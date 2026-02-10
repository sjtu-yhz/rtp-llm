#include "rtp_llm/cpp/models/logits_processor/TreeLogitsProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

using namespace std;

namespace rtp_llm {

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device) {};

TreeLogitsProcessor::TreeLogitsProcessor(rtp_llm::DeviceBase* device, std::vector<StreamTreeInfo> tree_infos):
    BaseLogitsProcessor(device), tree_infos_(tree_infos) {}

void TreeLogitsProcessor::generateVocabMask(
    size_t batch_size, size_t vocab_size, const std::vector<std::vector<size_t>>& batch_candidate_token_ids,cudaStream_t& stream,uint8_t* vocab_mask_pinned) {
    RTP_LLM_CHECK(batch_candidate_token_ids.size() == batch_size);
    std::fill(vocab_mask_pinned, vocab_mask_pinned + batch_size * vocab_size, 1);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto& candidate_token_ids = batch_candidate_token_ids[batch_idx];
        for (const auto& token_id : candidate_token_ids) {
            if (token_id < vocab_size) {
                vocab_mask_pinned[batch_idx * vocab_size + token_id] = 0;
            }
        }
    }

    BufferPtr host_buffer = std::make_shared<Buffer>(
        MemoryType::MEMORY_CPU_PINNED,
        DataType::TYPE_UINT8,
        std::vector<size_t>{batch_size * vocab_size},
        vocab_mask_pinned,
        nullptr
    );
    auto buffer_reshape = host_buffer->reshape({batch_size, vocab_size});
    //BufferPtr vocab_mask_buffer_cpu = vector2Buffer(vocab_mask_cpu);
    //auto      buffer_reshape        = vocab_mask_buffer_cpu->reshape({batch_size, vocab_size});
    //return device_->clone({buffer_reshape, rtp_llm::AllocationType::DEVICE});
    if (auto* cuda_device_ptr = dynamic_cast<CudaDevice*>(device_)){
        CloneParams clone_params={buffer_reshape, rtp_llm::AllocationType::DEVICE};
        auto dst = cuda_device_ptr->allocateBufferLike(clone_params.input, clone_params.alloc_type, clone_params.hints);
        CopyParams copy_params={*dst, clone_params.input, clone_params.overlapped, DeviceStream::DEFAULT, clone_params.async};

        copy_params.check();
        const auto& copy_src = copy_params.src;
        const auto& copy_dst = copy_params.dst;
        cudaMemcpyKind copyType = cudaMemcpyHostToDevice;
        cudaMemcpyAsync(copy_dst.data(), copy_src.data(), copy_src.sizeBytes(), copyType, stream);
        check_cuda_error();
        // device_->noBlockCopy(copy_params);
        batch_vocab_mask = dst;
    } else {
        batch_vocab_mask = device_->clone({buffer_reshape, rtp_llm::AllocationType::DEVICE});
    }

}

// void TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx,uint8_t* vocab_mask_pinned) {
//     auto batch_size = size();
//     RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
//     bool                             need_process = false;
//     std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

//     for (size_t i = 0; i < size(); ++i) {
//         auto& info = tree_infos_[i];
//         if (!info.in_tree_mode) {
//             continue;
//         }
        
//         // 【新增】如果即将到达终止状态，提前退出树模式
//         if (info.dfa_ptr->isAboutToFinish()) {
//             info.in_tree_mode = false;
//             continue;
//         }

//         const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
//         batch_candidate_token_ids[i]    = candidate_token_ids;
//         if (candidate_token_ids.size() > 0) {
//             need_process = true;
//         }
//     }
//     // If no beams need processing, return early
//     if (!need_process) {
//         return;
//     }
//     auto   batch_logits     = inputs.logits->slice(start_idx, batch_size);
//     size_t vocab_size       = batch_logits->shape()[1];
//     auto   batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids,vocab_mask_pinned);
//     maskLogits(batch_logits, batch_vocab_mask);
// }

void TreeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    auto batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
    bool                             need_process = false;
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }
        // 【新增】如果即将到达终止状态，提前退出树模式
        if (info.dfa_ptr->isAboutToFinish()) {
            info.in_tree_mode = false;
            continue;
        }
        const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
        batch_candidate_token_ids[i]    = candidate_token_ids;
        if (candidate_token_ids.size() > 0) {
            need_process = true;
        }
    }
    // If no beams need processing, return early
    if (!need_process) {
        return;
    }

    auto   batch_logits     = inputs.logits->slice(start_idx, batch_size);
    size_t vocab_size       = batch_logits->shape()[1];
    auto   batch_vocab_mask = generateVocabMask(batch_size, vocab_size, batch_candidate_token_ids);
    maskLogits(batch_logits, batch_vocab_mask);
}

rtp_llm::BufferPtr TreeLogitsProcessor::generateVocabMask(
    size_t batch_size, size_t vocab_size, const std::vector<std::vector<size_t>>& batch_candidate_token_ids) {
    RTP_LLM_CHECK(batch_candidate_token_ids.size() == batch_size);
    std::vector<uint8_t> vocab_mask_cpu(batch_size * vocab_size, 1);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto& candidate_token_ids = batch_candidate_token_ids[batch_idx];
        for (const auto& token_id : candidate_token_ids) {
            if (token_id < vocab_size) {
                vocab_mask_cpu[batch_idx * vocab_size + token_id] = 0;
            }
        }
    }

    BufferPtr vocab_mask_buffer_cpu = vector2Buffer(vocab_mask_cpu);
    auto      buffer_reshape        = vocab_mask_buffer_cpu->reshape({batch_size, vocab_size});
    return device_->clone({buffer_reshape, rtp_llm::AllocationType::DEVICE});
}


void TreeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::vector<StreamTreeInfo> new_tree_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_tree_infos.push_back(tree_infos_[src_batch_idx].copy());
    }
    tree_infos_ = std::move(new_tree_infos);
}

void TreeLogitsProcessor::updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens->shape().size());
    RTP_LLM_CHECK(size() == new_tokens->shape()[0]);

    for (size_t i = 0; i < size(); i++) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode)
            continue;

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK(num_new_tokens == new_tokens->shape()[1]);
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = *(*new_tokens)[i].dataWithOffset<int>(j + offset);
            // 开启软约束模式
            if (info.soft_constraint_mode) {
                // 先验证next token是否符合约束解码要求
                if (!info.dfa_ptr->isValidNext(current_token_id)) {
                    // 如果开启软约束模式，且next token不符合约束解码要求，则退出约束解码
                    RTP_LLM_LOG_WARNING("Soft constraint mode: Invalid token %d detected for beam %zu, "
                                      "exiting constraint decoding gracefully. Current status: %s",
                                      current_token_id, i, info.dfa_ptr->status().c_str());
                    info.in_tree_mode = false;
                    break;
                }
            }
            
            // 处理next token
            try {
                info.dfa_ptr->next(current_token_id);
            } catch (const std::runtime_error& e) {
                if (info.soft_constraint_mode) {
                    // 如果是软约束模式下，理论上走不到这个分支，保留用于兜底
                    RTP_LLM_LOG_WARNING("Soft constraint mode: Unexpected error processing token %d for beam %zu: %s",
                                      current_token_id, i, e.what());
                    info.in_tree_mode = false;
                    break;
                } else {
                    // 在严格模式下抛异常
                    throw;
                }
            }
        }

        info.current_output_length += num_new_tokens;
        // 【新增】检查 DFA 是否完成，如果完成则退出树模式，
        // 作为isAboutToFinish逻辑的兜底，确保即使isAboutToFinish()没有正确触发，DFA完成后也能退出树模式
        if (info.dfa_ptr->isFinished()) {
            info.in_tree_mode = false;
        }
    }
}

std::vector<std::vector<size_t>> TreeLogitsProcessor::getCandidateTokenIds(size_t start_idx, size_t finish_idx) {
    auto batch_size = size();
    RTP_LLM_CHECK(batch_size == finish_idx - start_idx);
    std::vector<std::vector<size_t>> batch_candidate_token_ids(batch_size);

    for (size_t i = 0; i < size(); ++i) {
        auto& info = tree_infos_[i];
        if (!info.in_tree_mode) {
            continue;
        }
        const auto& candidate_token_ids = info.dfa_ptr->getCandidateTokenIds();
        batch_candidate_token_ids[i]    = candidate_token_ids;
    }
    // If no beams need processing, return early
    return batch_candidate_token_ids;
}

TreeLogitsProcessorPtr TreeLogitsProcessor::fromGenerateInput(rtp_llm::DeviceBase*           device,
                                                              std::shared_ptr<GenerateInput> generate_input,
                                                              int32_t                        num) {
    if (!PrefixToCandidateTokens::instance()->initSuccess()) {
        return nullptr;
    }

    auto processor_ptr = std::make_shared<TreeLogitsProcessor>(rtp_llm::DeviceFactory::getDefaultDevice());

    // 读取软约束模式是否开启
    bool soft_constraint_enabled = generate_input->generate_config->soft_constraint_mode;
    
    if (soft_constraint_enabled) {
        RTP_LLM_LOG_INFO("TreeLogitsProcessor: Soft constraint mode enabled for %d streams", num);
    }

    for (size_t i = 0; i < num; i++) {
        StreamTreeInfo              tree_info(PrefixToCandidateTokens::instance()->initSuccess(),
                                 generate_input->inputLength(),
                                 0,
                                 generate_input->generate_config->hasNumBeams()
                                     || generate_input->generate_config->num_return_sequences > 1,
                                 std::make_shared<TreeDFA<std::string, int>>(PrefixToCandidateTokens::instance()),
                                 soft_constraint_enabled);
        std::vector<StreamTreeInfo> tree_infos       = {tree_info};
        auto                        single_processor = std::make_shared<TreeLogitsProcessor>(device, tree_infos);

        processor_ptr->insert(single_processor, 1);
    }

    return processor_ptr;
}

std::vector<std::string> TreeLogitsProcessor::getStatus() {
    std::vector<std::string> status_list;
    for (const auto& tree_info : tree_infos_) {
        status_list.push_back(tree_info.dfa_ptr->status());
    }
    return status_list;
}

}  // namespace rtp_llm
