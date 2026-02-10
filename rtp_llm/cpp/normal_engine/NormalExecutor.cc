#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include <cstdlib>
#include <memory>
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/models/PyWrappedModel.h"
#include "rtp_llm/cpp/models/NativeDeviceGraphModel.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"

using namespace std;

namespace rtp_llm {

NormalExecutor::NormalExecutor(const EngineInitParams&                   params,
                               const std::shared_ptr<CacheManager>&      cache_manager,
                               rtp_llm::DeviceBase*                      device,
                               std::shared_ptr<autil::LockFreeThreadPool> thread_pool,
                               const std::shared_ptr<lora::LoraManager>& lora_manager,
                               bool                                      warm_up):
    Executor(device),
    cache_manager_(cache_manager),
    lora_manager_(lora_manager),
    thread_pool_(std::move(thread_pool)),
    warm_up_(warm_up),
    use_all_gather_(params.gpt_init_parameter.use_all_gather_),
    metrics_reporter_(params.metrics_reporter),
    tps_reporter_(MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector>(metrics_reporter_)) {
    auto& gpt_param    = params.gpt_init_parameter;
    enable_detail_log_ = gpt_param.profiling_debug_logging_config.enable_detail_log;
    RTP_LLM_LOG_INFO("enable_detail_log_ = %d", enable_detail_log_);

    if (gpt_param.enable_eplb_ && gpt_param.moe_style_ != 0) {
        // use first moe layer weight as moe weight type
        int  first_moe_layer = gpt_param.moe_layer_index_.front();
        auto moe_weight_type = params.gpt_weights.layers[first_moe_layer].ffn_weights.moe_gate_weight->kernel->type();

        expert_balancer_ = make_shared<ExpertBalancer>(gpt_param.expert_num_,
                                                       gpt_param.phy_exp_num_,
                                                       gpt_param.num_layers_,
                                                       gpt_param.moe_inter_padding_size_,
                                                       gpt_param.hidden_size_,
                                                       gpt_param.eplb_update_time_,
                                                       gpt_param.ep_rank_,
                                                       gpt_param.ep_size_,
                                                       gpt_param.py_eplb_,
                                                       moe_weight_type,
                                                       device_,
                                                       gpt_param.eplb_mode_,
                                                       gpt_param.quant_algo_,
                                                       metrics_reporter_);
    }

    int               eos_id = params.gpt_init_parameter.special_tokens_.eos_token_id_;
    SamplerInitParams sampler_params{
        device_,
        eos_id,
        device->initParams().max_batch_size};  // set static max batch size to avoid sampler reset memory
    sampler_.reset(new Sampler(sampler_params));

    GptModelInitParams model_init_params(
        {device_,
         params.gpt_weights,
         genModelDescription(params.gpt_init_parameter),
         cache_manager ? ((optional<KVCacheAllocator::KVCacheBuffer>)cache_manager->kvCacheBuffer()) : nullopt,
         params.model_id});

    if (params.gpt_init_parameter.ffn_disaggregate_config.enable_ffn_disaggregate) {
        RTP_LLM_LOG_INFO("using ffn as service");
        enable_ffn_disaggregate_ = true;
    }
    if (!params.py_model.is_none()) {
        RTP_LLM_LOG_INFO("init executor with python model");
        model_.reset(new PyWrappedModel(model_init_params, params.py_model));
    } else if (device_->initParams().hw_kernel_config.enable_native_cuda_graph) {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model with native cuda graph");
        model_.reset(new NativeDeviceGraphModel(model_init_params));
    } else {
        RTP_LLM_LOG_INFO("init legacy c++ gpt model");
        model_.reset(new GptModel(model_init_params));
    }

    // when warmup, cache manager maybe nullptr
    const auto& cache_config = cache_manager ? cache_manager->cacheConfig() : CacheConfig();
    batch_stream_processor_.reset(
        new NormalBatchStreamProcessor(params.gpt_init_parameter, cache_config, thread_pool_, warm_up_));
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(
        params.gpt_init_parameter.ckpt_path_, params.gpt_init_parameter.sp_config.tree_decode_config);
    device_->profileStart();
    if (auto* cuda_device_ptr = dynamic_cast<CudaDevice*>(device_)){
        MAX_CUDA_MALLOC_SIZE = 200*170000;
        cudaError_t err = cudaMallocHost(&vocab_mask_pinned_ptr, MAX_CUDA_MALLOC_SIZE * sizeof(uint8_t));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory");
        }
    }

    cudaError_t err = cudaStreamCreate(&stream_comm);
    if (err != cudaSuccess) {
        printf("Failed to create stream_comm: %s\n", cudaGetErrorString(err));
    }
}

absl::Status NormalExecutor::process(const std::list<GenerateStreamPtr>& streams) {
    StreamGroups                   stream_groups(streams);
    RtpLLMExecutorMetricsCollector executor_collector;
    RtpLLMTokenPSMetricsCollector  tps_collector;
    GptModelInputs                 model_input;
    GptModelOutputs                model_output;
    SamplerOutput                  sampler_output;
    {
        int64_t start_time_us      = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    model_input_status = batch_stream_processor_->gatherModelInput(stream_groups);
        RETURN_IF_STATUS_OR_ERROR(model_input_status);
        model_input                              = std::move(model_input_status.value());
        executor_collector.gather_model_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        model_input.skip_run  = streams.empty() && !enable_ffn_disaggregate_;
        tpSyncModelInputs(model_input, device_);
        if (model_input.skip_run) {
            return absl::OkStatus();
        }
        executor_collector.tp_sync_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        // update kv cache
        if (model_input.kv_cache_update_mapping) {
            cache_manager_->blockBatchCopy(*model_input.kv_cache_update_mapping);
        }
    }
    // get lora input
    if (lora_manager_) {
        model_input.lora_model_input =
            lora_manager_->makeLoraModelInput(model_input.lora_ids, model_input.lora_input_lengths);
    }
    //*********** get LogitsProcessorStatesPtr begin ************//
    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
    auto all_streams          = stream_groups.allStreams();
    std::for_each(all_streams.begin(), all_streams.end(), [&state_ptr, idx = 0](auto& stream) mutable {
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            state_ptr->insert(processor, idx, idx + stream->currentBatchSize());
        }
        idx += stream->currentBatchSize();
    });
    bool need_process=false;
    size_t  bs;
    //*********** get LogitsProcessorStatesPtr end ************//
    {
        bool force = device_->getDeviceProperties().tp_rank == 0 && enable_detail_log_;
        if (force) {
            RTP_LLM_LOG_INFO("model_input: %s", model_input.debugString(force).c_str());
        } else {
            RTP_LLM_LOG_TRACE("model_input: %s", model_input.debugString(force).c_str());
        }
        int64_t start_time_us               = autil::TimeUtility::currentTimeInMicroSeconds();
        model_->forward_one_stage(model_input);
        if (state_ptr != nullptr) {
            auto logits_processors_ori = state_ptr->logits_processors_;
            auto intervals_ori = state_ptr->intervals_;
            for (size_t k = 0; k < logits_processors_ori.size(); k++) {
                if (auto* ptr_ori = dynamic_cast<TreeLogitsProcessor*>(logits_processors_ori[k].get())) {
                    std::vector<std::vector<size_t>> batch_candidate_token_ids =ptr_ori->getCandidateTokenIds(intervals_ori[k].first, intervals_ori[k].second);                   
                    for (size_t kth = 0; kth < ptr_ori->size(); ++kth) {
                        if (batch_candidate_token_ids[kth].size() > 0) {
                            need_process = true;
                            break; // Early exit when we find at least one non-empty candidate token list
                        }
                    }
                    if (need_process){
                        // Will be initialized after model forward pass with correct vocab_size
                        size_t vocab_size=31872; 
                        bs = ptr_ori->size();
                        ptr_ori->generateVocabMask(ptr_ori->size(), vocab_size, batch_candidate_token_ids,stream_comm,vocab_mask_pinned_ptr);
                    }
                }
            }
        }
        model_->forward_two_stage(model_input);
        model_output = std::move(model_->forward_three_stage(model_input));
        //model_output                        = std::move(model_->forward(model_input));
        executor_collector.model_forward_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        RTP_LLM_LOG_DEBUG("model forward done");
    }
    if (expert_balancer_) {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        expert_balancer_->stepForward(*model_, executor_collector);
        executor_collector.eplb_step_latency_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    if (device_->getDeviceProperties().tp_rank > 0 || warm_up_ || streams.size() == 0) {
        return absl::OkStatus();
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        CHECK_AND_RETURN_REF(sampler_input,
                             batch_stream_processor_->gatherSamplerInput(stream_groups, model_input, model_output));

        // Initialize vocab_size from sampler input logits shape
        size_t vocab_size = 0;
        if (sampler_input.logits != nullptr && sampler_input.logits->shape().size() > 1) {
            vocab_size = sampler_input.logits->shape()[1];
        }
        int64_t sample_start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        /******** logits processor begin ********/
        if (sampler_input.logits_processor_states_ptr != nullptr) {
            // sampler_input.logits_processor_states_ptr->batchProcess(sampler_input);
            auto logits_processors_list = sampler_input.logits_processor_states_ptr->logits_processors_;
            auto intervals_list = sampler_input.logits_processor_states_ptr->intervals_;
            for (size_t ith = 0; ith < logits_processors_list.size(); ith++) {
                if (auto* ptr = dynamic_cast<TreeLogitsProcessor*>(logits_processors_list[ith].get())) {
                    if (need_process && ptr->batch_vocab_mask != nullptr){
                       auto batch_logits = sampler_input.logits->slice(intervals_list[ith].first, intervals_list[ith].second);
                        ptr->maskLogits(batch_logits,ptr->batch_vocab_mask);  
                    }
                    //ptr->process(sampler_input, intervals_list[ith].first, intervals_list[ith].second,vocab_mask_pinned_ptr);
                } else if (auto* ptr = dynamic_cast<ThinkModeLogitsProcessor*>(logits_processors_list[ith].get())){
                    ptr->process(sampler_input, intervals_list[ith].first, intervals_list[ith].second);
                } else if (auto* ptr = dynamic_cast<MultiSeqLogitsProcessor*>(logits_processors_list[ith].get())){
                    ptr->process(sampler_input, intervals_list[ith].first, intervals_list[ith].second);
                }
            }
        }
        /******** logits processor end ********/
        sampler_output = std::move(sampler_->forward(sampler_input));
        RTP_LLM_LOG_DEBUG("sampler forward done");
        executor_collector.sample_input_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
    }
    {
        int64_t start_time_us = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    result =
            batch_stream_processor_->dispatch(stream_groups, {std::move(model_output), std::move(sampler_output)});
        executor_collector.dispatch_output_us = autil::TimeUtility::currentTimeInMicroSeconds() - start_time_us;
        reportMetrics(stream_groups, executor_collector, tps_collector);
        return result;
    }
}

void NormalExecutor::reportMetrics(const StreamGroups&             stream_groups,
                                   RtpLLMExecutorMetricsCollector& executor_collector,
                                   RtpLLMTokenPSMetricsCollector&  tps_collector) {
    if (device_->getDeviceProperties().tp_rank > 0) {
        return;
    }
    if (metrics_reporter_) {
        executor_collector.context_batch_size  = stream_groups.totalContextBatchSize();
        executor_collector.generate_batch_size = stream_groups.totalDecodeBatchSize();
        executor_collector.execute_token_size  = stream_groups.modelExecuteTokenSize();
        executor_collector.max_seq_len         = stream_groups.maxSeqLen();
        if (executor_collector.context_batch_size != 0) {
            executor_collector.context_batch_size_when_has_context  = executor_collector.context_batch_size;
            executor_collector.generate_batch_size_when_has_context = executor_collector.generate_batch_size;
            executor_collector.execute_token_size_when_has_context  = executor_collector.execute_token_size;
            executor_collector.max_seq_len_when_has_context         = executor_collector.max_seq_len;
        }
        metrics_reporter_->report<RtpLLMExecutorMetrics, RtpLLMExecutorMetricsCollector>(nullptr, &executor_collector);

        tps_collector.context_tps  = stream_groups.modelExecuteTokenSize() - stream_groups.totalDecodeBatchSize();
        tps_collector.generate_tps = stream_groups.totalDecodeBatchSize();
        tps_collector.total_tps    = stream_groups.modelExecuteTokenSize();
        tps_reporter_.report(&tps_collector);
    }
}

bool NormalExecutor::updateEplbConfig(const EplbConfig& config) {
    if (expert_balancer_) {
        return expert_balancer_->updateEplbConfig(config);
    }
    return true;
}

}  // namespace rtp_llm
