// Copyright © 2024 Apple Inc.

import CoreImage
import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Model Configuration

public struct Magistral3Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let headDim: Int
        public let intermediateSize: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let maxPositionEmbeddings: Int
        public let vocabSize: Int
        public let hiddenAct: String
        private let _attentionDropout: Float?
        public var attentionDropout: Float { _attentionDropout ?? 0.0 }
        private let _slidingWindow: Int?
        public var slidingWindow: Int? { _slidingWindow }
        private let _useCache: Bool?
        public var useCache: Bool { _useCache ?? true }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case vocabSize = "vocab_size"
            case hiddenAct = "hidden_act"
            case _attentionDropout = "attention_dropout"
            case _slidingWindow = "sliding_window"
            case _useCache = "use_cache"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let headDim: Int
        public let intermediateSize: Int
        public let imageSize: Int
        public let patchSize: Int
        public let numChannels: Int
        public let ropeTheta: Float
        public let hiddenAct: String
        private let _attentionDropout: Float?
        public var attentionDropout: Float { _attentionDropout ?? 0.0 }
        private let _initializer_range: Float?
        public var initializerRange: Float { _initializer_range ?? 0.02 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case numChannels = "num_channels"
            case ropeTheta = "rope_theta"
            case hiddenAct = "hidden_act"
            case _attentionDropout = "attention_dropout"
            case _initializer_range = "initializer_range"
        }
    }

    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration
    public let modelType: String
    public let imageTokenIndex: Int
    public let spatialMergeSize: Int
    public let projectorHiddenAct: String
    public let multimodalProjectorBias: Bool
    private let _visionFeatureLayer: Int?
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfig.vocabSize }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case imageTokenIndex = "image_token_index"
        case spatialMergeSize = "spatial_merge_size"
        case projectorHiddenAct = "projector_hidden_act"
        case multimodalProjectorBias = "multimodal_projector_bias"
        case _visionFeatureLayer = "vision_feature_layer"
        case _vocabSize = "vocab_size"
    }
}

// MARK: - Processor Configuration

public struct Magistral3ProcessorConfiguration: Codable, Sendable {

    public struct Size: Codable, Sendable {
        public let longestEdge: Int?
        public let width: Int?
        public let height: Int?

        var cgSize: CGSize {
            if let width = width, let height = height {
                return .init(width: width, height: height)
            } else if let edge = longestEdge {
                return .init(width: edge, height: edge)
            }
            return .init(width: 1540, height: 1540)
        }

        enum CodingKeys: String, CodingKey {
            case longestEdge = "longest_edge"
            case width
            case height
        }
    }

    public struct PatchSize: Codable, Sendable {
        public let width: Int?
        public let height: Int?

        var size: Int {
            width ?? height ?? 14
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size?
    public let patchSize: PatchSize?
    // These fields come from model config.json, not preprocessor_config.json
    private let _spatialMergeSize: Int?
    private let _imageTokenIndex: Int?

    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? 10 }
    public var actualPatchSize: Int { patchSize?.size ?? 14 }
    public var actualSize: CGSize { size?.cgSize ?? .init(width: 1540, height: 1540) }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case patchSize = "patch_size"
        case _spatialMergeSize = "spatial_merge_size"
        case _imageTokenIndex = "image_token_index"
    }

    // Default values if preprocessor_config.json is missing
    public init(patchSize: Int = 14, spatialMergeSize: Int = 2, imageTokenIndex: Int = 10) {
        // ImageNet normalization as default for Pixtral
        self.imageMean = [0.48145466, 0.4578275, 0.40821073]
        self.imageStd = [0.26862954, 0.26130258, 0.27577711]
        self.size = nil
        self.patchSize = nil
        self._spatialMergeSize = spatialMergeSize
        self._imageTokenIndex = imageTokenIndex
    }
}

// MARK: - Vision Model Components

private enum Vision {

    // Rotary position embedding for vision transformer
    final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq =
                1.0
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                        / Float(dimension)
                )
            let seq = MLXArray(0 ..< sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    // Patch embeddings for vision transformer
    final class PatchEmbeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_conv") var patchEmbedding: Conv2d

        let patchSize: Int
        let hiddenSize: Int

        init(config: Magistral3Configuration.VisionConfiguration) {
            self.patchSize = config.patchSize
            self.hiddenSize = config.hiddenSize

            self._patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            // x shape: [B, C, H, W]
            // output shape: [B, num_patches_h, num_patches_w, hidden_size]
            var patches = patchEmbedding(x)
            // Conv2d output is [B, hidden_size, H', W'], need to transpose
            patches = patches.transposed(0, 2, 3, 1)
            return patches
        }
    }

    // Vision attention module
    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(config: Magistral3Configuration.VisionConfiguration) {
            self.numHeads = config.numAttentionHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            let dim = config.hiddenSize
            self._qProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
            self._kProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(dim, numHeads * headDim, bias: false)
            self._oProj.wrappedValue = Linear(numHeads * headDim, dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray, rotaryPosEmb: MLXArray) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

            // Apply rotary embeddings
            queries = Vision.applyRotary(queries, freqs: rotaryPosEmb)
            keys = Vision.applyRotary(keys, freqs: rotaryPosEmb)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return oProj(output)
        }
    }

    // Vision MLP module - gated architecture
    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(config: Magistral3Configuration.VisionConfiguration) {
            self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    // Vision transformer block
    final class EncoderLayer: Module {
        @ModuleInfo(key: "attention") var attention: Attention
        @ModuleInfo(key: "feed_forward") var mlp: MLP
        @ModuleInfo(key: "attention_norm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var postAttentionLayerNorm: RMSNorm

        init(config: Magistral3Configuration.VisionConfiguration) {
            self._attention.wrappedValue = Attention(config: config)
            self._mlp.wrappedValue = MLP(config: config)
            self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-5)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-5)
        }

        func callAsFunction(_ x: MLXArray, rotaryPosEmb: MLXArray) -> MLXArray {
            var h = x + attention(inputLayerNorm(x), rotaryPosEmb: rotaryPosEmb)
            h = h + mlp(postAttentionLayerNorm(h))
            return h
        }
    }

    // Transformer wrapper for layers
    final class Transformer: Module {
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(config: Magistral3Configuration.VisionConfiguration) {
            self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
                EncoderLayer(config: config)
            }
        }
    }

    // Inner vision model (matches Python's PixtralVisionModel)
    final class PixtralVisionModelCore: Module {
        @ModuleInfo(key: "patch_conv") var patchConv: Conv2d
        @ModuleInfo(key: "transformer") var transformer: Transformer
        @ModuleInfo(key: "ln_pre") var postLayerNorm: RMSNorm

        let config: Magistral3Configuration.VisionConfiguration
        let rotaryEmbedding: VisionRotaryEmbedding

        init(_ config: Magistral3Configuration.VisionConfiguration) {
            self.config = config

            self._patchConv.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false
            )
            self._transformer.wrappedValue = Transformer(config: config)
            self._postLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-5)

            let headDim = config.headDim
            self.rotaryEmbedding = VisionRotaryEmbedding(dimension: headDim / 2, theta: config.ropeTheta)
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            // pixelValues: [B, C, H, W]
            // Apply patch convolution: [B, C, H, W] -> [B, hidden_size, H', W']
            var hiddenStates = patchConv(pixelValues)
            // Transpose to [B, H', W', hidden_size]
            hiddenStates = hiddenStates.transposed(0, 2, 3, 1)

            // Flatten spatial dimensions
            let (B, H, W, D) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2), hiddenStates.dim(3))
            hiddenStates = hiddenStates.reshaped(B, H * W, D)

            // Compute rotary embeddings
            let seqLen = H * W
            let freqs = rotaryEmbedding(sequenceLength: Int(sqrt(Double(seqLen))))

            // Prepare 2D positional embeddings for all patches
            let patchH = H
            let patchW = W

            var hIndices = MLXArray(0 ..< patchH).asType(.int32)
            hIndices = expandedDimensions(hIndices, axis: 1)
            hIndices = broadcast(hIndices, to: [patchH, patchW])

            var wIndices = MLXArray(0 ..< patchW).asType(.int32)
            wIndices = expandedDimensions(wIndices, axis: 0)
            wIndices = broadcast(wIndices, to: [patchH, patchW])

            let hFreqs = freqs[hIndices.flattened(), 0...]
            let wFreqs = freqs[wIndices.flattened(), 0...]
            let rotaryPosEmb = concatenated([hFreqs, wFreqs], axis: -1)

            // Apply transformer layers
            for layer in transformer.layers {
                hiddenStates = layer(hiddenStates, rotaryPosEmb: rotaryPosEmb)
            }

            hiddenStates = postLayerNorm(hiddenStates)

            return hiddenStates
        }
    }

    // Wrapper class that adds the vision_model layer (matches Python's VisionModel)
    final class PixtralVisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: PixtralVisionModelCore

        init(_ config: Magistral3Configuration.VisionConfiguration) {
            self._visionModel.wrappedValue = PixtralVisionModelCore(config)
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            return visionModel(pixelValues)
        }
    }
}

// MARK: - Language Model Components

private enum Language {

    // Mistral3 attention with grouped-query attention
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: Magistral3Configuration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self._rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            if let cache {
                queries = rotaryEmbedding(queries, offset: cache.offset)
                keys = rotaryEmbedding(keys, offset: cache.offset)
            } else {
                queries = rotaryEmbedding(queries)
                keys = rotaryEmbedding(keys)
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return wo(output)
        }
    }

    // Mistral3 MLP with SiLU activation
    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(_ config: Magistral3Configuration.TextConfiguration) {
            self._gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            self._down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
            self._up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    // Mistral3 decoder layer
    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Magistral3Configuration.TextConfiguration) {
            self._attention.wrappedValue = Attention(config)
            self._mlp.wrappedValue = MLP(config)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            let out = h + r
            return out
        }
    }

    // Mistral3 model
    final class Mistral3Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Magistral3Configuration.TextConfiguration) {
            precondition(config.vocabSize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize)

            self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
                DecoderLayer(config)
            }
            self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: h, cache: cache)

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return norm(h)
        }
    }

    // Language model wrapper
    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo(key: "model") var model: Mistral3Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]

        init(_ config: Magistral3Configuration.TextConfiguration) {
            self._model.wrappedValue = Mistral3Model(config)

            // Mistral3 typically uses tied embeddings, but check config
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)

            self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil
        ) -> LMOutput {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Multimodal Projector

private class PatchMerger: Module {
    @ModuleInfo(key: "merging_layer") var mergingLayer: Linear
    let spatialMergeSize: Int

    init(hiddenSize: Int, spatialMergeSize: Int) {
        self.spatialMergeSize = spatialMergeSize
        self._mergingLayer.wrappedValue = Linear(
            hiddenSize * spatialMergeSize * spatialMergeSize,
            hiddenSize,
            bias: false
        )
    }

    func callAsFunction(_ x: MLXArray, imageSizes: [IntOrPair]) -> MLXArray {
        // For now, simple implementation - assumes already processed
        // Full implementation would need unfold operation
        mergingLayer(x)
    }
}

private class MultimodalProjector: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") var patchMerger: PatchMerger
    @ModuleInfo(key: "linear_1") var linear1: Linear
    let gelu: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(visionHiddenSize: Int, textHiddenSize: Int, spatialMergeSize: Int, bias: Bool) {
        self._norm.wrappedValue = RMSNorm(dimensions: visionHiddenSize)
        self._patchMerger.wrappedValue = PatchMerger(
            hiddenSize: visionHiddenSize,
            spatialMergeSize: spatialMergeSize
        )
        self._linear1.wrappedValue = Linear(visionHiddenSize, textHiddenSize, bias: bias)
        self.gelu = GELU()
        self._linear2.wrappedValue = Linear(textHiddenSize, textHiddenSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray, imageSizes: [IntOrPair]) -> MLXArray {
        var h = norm(x)
        h = patchMerger(h, imageSizes: imageSizes)
        h = linear1(h)
        h = gelu(h)
        h = linear2(h)
        return h
    }
}

// MARK: - Main VLM Model

public class Magistral3ForConditionalGeneration: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") fileprivate var visionModel: Vision.PixtralVisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel: Language.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") fileprivate var multiModalProjector: MultimodalProjector

    public let config: Magistral3Configuration

    public var vocabularySize: Int { config.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var headDim: MLX.IntOrPair { .init(config.textConfig.headDim) }

    public init(_ config: Magistral3Configuration) {
        self.config = config
        self._visionModel.wrappedValue = Vision.PixtralVisionModel(config.visionConfig)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfig)
        self._multiModalProjector.wrappedValue = MultimodalProjector(
            visionHiddenSize: config.visionConfig.hiddenSize,
            textHiddenSize: config.textConfig.hiddenSize,
            spatialMergeSize: config.spatialMergeSize,
            bias: config.multimodalProjectorBias
        )
    }

    private func mergeSpatialPatches(_ hiddenStates: MLXArray, mergeSize: Int) -> MLXArray {
        // hiddenStates: [B, num_patches, hidden_size]
        let (B, numPatches, D) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))

        // Calculate grid dimensions
        let gridSize = Int(sqrt(Double(numPatches)))
        guard gridSize * gridSize == numPatches else {
            // If not a perfect square, return as-is
            return hiddenStates
        }

        // Reshape to grid
        var states = hiddenStates.reshaped(B, gridSize, gridSize, D)

        // Spatial merge: combine mergeSize x mergeSize patches
        let newGridSize = gridSize / mergeSize
        guard newGridSize > 0 else {
            return hiddenStates
        }

        // Reshape and merge
        states = states.reshaped(B, newGridSize, mergeSize, newGridSize, mergeSize, D)
        states = states.transposed(0, 1, 3, 2, 4, 5)
        states = states.reshaped(B, newGridSize * newGridSize, mergeSize * mergeSize * D)

        return states
    }

    private func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues = pixelValues else {
            // Text-only mode
            return languageModel.model.embedTokens(inputIds)
        }

        // Extract vision features
        var visionFeatures = visionModel(pixelValues)

        // Apply spatial merging
        if config.spatialMergeSize > 1 {
            visionFeatures = mergeSpatialPatches(visionFeatures, mergeSize: config.spatialMergeSize)
        }

        // Project to text embedding space
        // TODO: Pass actual image sizes from input
        let imageSizes: [IntOrPair] = []
        visionFeatures = multiModalProjector(visionFeatures, imageSizes: imageSizes)

        // Get text embeddings
        let embeddings = languageModel.model.embedTokens(inputIds)

        // Replace image tokens with vision features following Idefics3 pattern
        // Assumes batch size == 1
        // asArray returns a flat 1D array - inputIds [1, seq_len] -> [seq_len] values
        let allIds = inputIds.asArray(Int.self)
        let imageTokenId = config.imageTokenIndex

        // Find all positions where image tokens occur
        let imagePositions = allIds.enumerated().compactMap { idx, tokenId in
            tokenId == imageTokenId ? idx : nil
        }

        // Build final embeddings by concatenating text and vision segments
        var segments = [MLXArray]()
        var startIdx = 0

        // Vision features: (B=1, num_patches, hidden_size)
        let numVisionTokens = visionFeatures.dim(1)

        for (visionIdx, pos) in imagePositions.enumerated() {
            // Add text embeddings before this image token
            if pos > startIdx {
                segments.append(embeddings[0, startIdx..<pos])
            }

            // Add vision feature for this position
            if visionIdx < numVisionTokens {
                segments.append(visionFeatures[0, visionIdx..<visionIdx + 1])
            }

            startIdx = pos + 1
        }

        // Add remaining text embeddings after last image token
        if startIdx < embeddings.dim(1) {
            segments.append(embeddings[0, startIdx...])
        }

        let finalEmbeddings = concatenated(segments, axis: 0)
        return finalEmbeddings.expandedDimensions(axis: 0)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let image = input.image else {
            // Text-only inference
            let result = languageModel(input.text.tokens, cache: cache)
            return .logits(result)
        }

        let inputIds = input.text.tokens
        let pixelValues = image.pixels

        let inputEmbedding = inputEmbeddings(inputIds: inputIds, pixelValues: pixelValues)

        let result = languageModel(nil, cache: cache, inputEmbedding: inputEmbedding)

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Keys to filter with contains check
        let containsFilters = [
            "self_attn.rotary_emb.inv_freq",
            "rotary_emb.inv_freq"
        ]

        // Exact keys to filter (unqualified top-level keys that don't match our model structure)
        let exactFilters: Set<String> = [
            "layers", "norm", "output", "patch_merger", "pre_mm_projector_norm",
            "tok_embeddings", "vision_encoder", "vision_language_adapter"
        ]

        let sanitized = weights.filter { key, _ in
            // Filter out keys that contain certain strings
            let hasContainsFilter = containsFilters.contains(where: { key.contains($0) })
            // Filter out exact key matches
            let isExactFilter = exactFilters.contains(key)

            return !hasContainsFilter && !isExactFilter
        }

        // Note: MLX-converted models already have patch_conv weights in correct format [O, H, W, I]
        // The key is vision_tower.vision_model.patch_conv.weight (not vision_tower.vision_model.vision_model.patch_conv.weight)
        // No transposition needed for MLX-format models

        return sanitized
    }
}

extension Magistral3ForConditionalGeneration: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

// MARK: - Processor

public final class Magistral3Processor: UserInputProcessor {

    private let config: Magistral3ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Magistral3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func preprocessImage(_ image: CIImage, processing: UserInput.Processing?) -> MLXArray {
        var processedImage = image

        // Ensure sRGB tone curve
        processedImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)

        // Apply user processing
        processedImage = MediaProcessing.apply(processedImage, processing: processing)

        // Resize to target size
        processedImage = MediaProcessing.resampleBicubic(processedImage, to: config.actualSize)

        // Normalize with ImageNet statistics
        processedImage = MediaProcessing.normalize(
            processedImage,
            mean: config.imageMeanTuple,
            std: config.imageStdTuple
        )

        return MediaProcessing.asMLXArray(processedImage)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Generate messages from input
        let messages = DefaultMessageGenerator().generate(from: input)

        // Apply chat template
        let promptTokens = try tokenizer.applyChatTemplate(messages: messages)

        if input.images.isEmpty {
            // Text-only mode
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask))
        }

        // Single image support for now
        guard input.images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        // Preprocess image
        let pixels = try preprocessImage(input.images[0].asCIImage(), processing: input.processing)

        // Calculate number of image tokens after spatial merging
        let imageSize = Int(config.actualSize.width)
        let patchSize = config.actualPatchSize
        let spatialMergeSize = config.spatialMergeSize
        let patchesPerSide = imageSize / patchSize
        let mergedPatchesPerSide = patchesPerSide / spatialMergeSize
        let numImageTokens = mergedPatchesPerSide * mergedPatchesPerSide

        // Insert image tokens by replacing image token index with actual tokens
        var tokenIds = promptTokens
        let imageTokenId = config.imageTokenIndex

        // Find the image token and replace it with the correct number of image tokens
        if let imageTokenIdx = tokenIds.firstIndex(of: imageTokenId) {
            // Replace single image token with numImageTokens copies
            tokenIds.remove(at: imageTokenIdx)
            tokenIds.insert(contentsOf: Array(repeating: imageTokenId, count: numImageTokens), at: imageTokenIdx)
        }

        let promptArray = MLXArray(tokenIds).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: .init(pixels: pixels.expandedDimensions(axis: 0))
        )
    }
}
