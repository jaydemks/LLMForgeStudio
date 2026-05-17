using LLMForgeStudio.App.Core.Dataset;
using LLMForgeStudio.App.Core.Generation;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;

namespace LLMForgeStudio.App.Core.Guidance;

public static class GuidedDefaultsEngine
{
    public static void ApplyTokenizerPreset(TokenizerKind kind, string datasetText, TokenizerConfig tokenizer)
    {
        var approxWords = Math.Max(1, TextCleaner.Analyze(datasetText).ApproxWordCount);

        switch (kind)
        {
            case TokenizerKind.Character:
                tokenizer.TargetVocabSize = 256;
                tokenizer.MinFrequency = 1;
                tokenizer.MaxMerges = 0;
                tokenizer.KeepPunctuationAsTokens = true;
                break;
            case TokenizerKind.Word:
                tokenizer.TargetVocabSize = Math.Clamp(approxWords / 3, 1_000, 30_000);
                tokenizer.MinFrequency = 2;
                tokenizer.MaxMerges = 0;
                tokenizer.KeepPunctuationAsTokens = true;
                break;
            case TokenizerKind.ByteLevelBpe:
                tokenizer.TargetVocabSize = Math.Clamp(approxWords / 4, 4_000, 24_000);
                tokenizer.MinFrequency = approxWords < 250_000 ? 1 : 2;
                tokenizer.MaxMerges = Math.Clamp((int)(tokenizer.TargetVocabSize * 0.75), 2_000, 32_000);
                tokenizer.KeepPunctuationAsTokens = true;
                break;
            case TokenizerKind.Unigram:
                tokenizer.TargetVocabSize = Math.Clamp(approxWords / 4, 2_000, 24_000);
                tokenizer.MinFrequency = 2;
                tokenizer.MaxMerges = 0;
                tokenizer.KeepPunctuationAsTokens = true;
                break;
            case TokenizerKind.WordPiece:
                tokenizer.TargetVocabSize = Math.Clamp(approxWords / 4, 2_000, 20_000);
                tokenizer.MinFrequency = 2;
                tokenizer.MaxMerges = 0;
                tokenizer.KeepPunctuationAsTokens = true;
                break;
            case TokenizerKind.SimpleBpe:
                tokenizer.TargetVocabSize = Math.Clamp(approxWords / 4, 3_000, 16_000);
                tokenizer.MinFrequency = approxWords < 250_000 ? 1 : 2;
                tokenizer.MaxMerges = Math.Clamp((int)(tokenizer.TargetVocabSize * 0.7), 1_500, 20_000);
                tokenizer.KeepPunctuationAsTokens = true;
                break;
            case TokenizerKind.HybridFallback:
                tokenizer.TargetVocabSize = Math.Clamp(approxWords / 4, 2_000, 20_000);
                tokenizer.MinFrequency = 2;
                tokenizer.MaxMerges = Math.Clamp(tokenizer.TargetVocabSize / 2, 800, 30_000);
                tokenizer.KeepPunctuationAsTokens = true;
                tokenizer.UseCharacterFallback = true;
                break;
            case TokenizerKind.HierarchicalExperimental:
                tokenizer.TargetVocabSize = Math.Clamp(approxWords / 5, 2_000, 24_000);
                tokenizer.MinFrequency = 2;
                tokenizer.MaxMerges = Math.Clamp(tokenizer.TargetVocabSize / 2, 1_000, 40_000);
                break;
        }
    }

    public static void ApplyModelTrainingPresetForTokenizer(TokenizerKind kind, ModelConfig model, TrainingConfig training)
    {
        switch (kind)
        {
            case TokenizerKind.Character:
                model.BlockSize = 256;
                model.Layers = 6;
                model.Heads = 6;
                model.EmbeddingSize = 384;
                training.BatchSize = 32;
                training.LearningRate = 3e-4;
                training.EvalEvery = 100;
                break;
            case TokenizerKind.ByteLevelBpe:
                // Safer default for from-scratch conversational quality on small/medium corpora.
                // Slightly smaller capacity + lower LR tends to reduce gibberish collapse.
                model.BlockSize = 192;
                model.Layers = 6;
                model.Heads = 6;
                model.EmbeddingSize = 384;
                training.BatchSize = 16;
                training.LearningRate = 1e-4;
                training.EvalEvery = 60;
                break;
            case TokenizerKind.Unigram:
                model.BlockSize = 224;
                model.Layers = 8;
                model.Heads = 8;
                model.EmbeddingSize = 512;
                training.BatchSize = 20;
                training.LearningRate = 2e-4;
                training.EvalEvery = 80;
                break;
            case TokenizerKind.WordPiece:
                model.BlockSize = 224;
                model.Layers = 8;
                model.Heads = 8;
                model.EmbeddingSize = 512;
                training.BatchSize = 20;
                training.LearningRate = 2e-4;
                training.EvalEvery = 80;
                break;
            case TokenizerKind.Word:
                model.BlockSize = 192;
                model.Layers = 6;
                model.Heads = 8;
                model.EmbeddingSize = 512;
                training.BatchSize = 24;
                training.LearningRate = 2.5e-4;
                training.EvalEvery = 80;
                break;
            case TokenizerKind.SimpleBpe:
                model.BlockSize = 192;
                model.Layers = 8;
                model.Heads = 8;
                model.EmbeddingSize = 512;
                training.BatchSize = 24;
                training.LearningRate = 2e-4;
                training.EvalEvery = 80;
                break;
            case TokenizerKind.HybridFallback:
                model.BlockSize = 224;
                model.Layers = 8;
                model.Heads = 8;
                model.EmbeddingSize = 512;
                training.BatchSize = 20;
                training.LearningRate = 2e-4;
                training.EvalEvery = 80;
                break;
            case TokenizerKind.HierarchicalExperimental:
                model.BlockSize = 160;
                model.Layers = 6;
                model.Heads = 6;
                model.EmbeddingSize = 384;
                training.BatchSize = 16;
                training.LearningRate = 2e-4;
                training.EvalEvery = 60;
                break;
        }

        training.MaxSteps = Math.Max(training.MaxSteps, 200);
    }

    public static void ApplyTrainingProfile(string profile, TrainingConfig training)
    {
        switch (profile)
        {
            case "Tiny":
                training.BatchSize = 8;
                training.MaxSteps = Math.Max(training.MaxSteps, 200);
                training.LearningRate = 4e-4;
                training.EvalEvery = 25;
                training.Optimizer = "adamw";
                training.Scheduler = "none";
                training.WarmupSteps = 0;
                training.MixedPrecision = false;
                training.Precision = "fp16";
                training.EnableGradientClipping = true;
                training.GradientClipNorm = 1.0;
                training.CheckpointEvery = 100;
                training.EnableDeduplication = false;
                training.RemoveDuplicateLines = false;
                training.RemoveDuplicateParagraphs = false;
                training.NormalizeUnicode = true;
                training.CollapseWhitespace = false;
                training.CurriculumLearning = false;
                training.CurriculumWarmupRatio = 0.2;
                training.DistributedTraining = false;
                training.OrchestratePipelineStages = false;
                training.MultiGpuStrategy = "none";
                training.GradientAccumulationSteps = 1;
                training.AutoDeviceMap = true;
                training.EnablePostTrainingQuantization = false;
                training.QuantizationProfile = "dynamic-int8";
                training.QuantizationCalibrationSamples = 64;
                training.EnableQatPath = false;
                training.QatFineTuneSteps = 100;
                training.EvalSuite = "basic";
                training.FineTuningOrchestration = false;
                training.FineTuneStageSft = true;
                training.FineTuneStageDpo = false;
                training.FineTuneStageRlhf = false;
                training.RlhfFeedbackSource = "inline";
                training.RlhfFeedbackPath = string.Empty;
                training.RewardModelingEnabled = false;
                training.SafetyPolicyMode = "standard";
                training.ExportOnnx = false;
                training.ExportGguf = false;
                break;
            case "Balanced":
                training.Optimizer = "adamw";
                training.Scheduler = "cosine";
                training.WarmupSteps = 100;
                training.MixedPrecision = true;
                training.Precision = "fp16";
                training.CheckpointEvery = 250;
                training.EvalSuite = "quick-5";
                training.DistributedTraining = false;
                training.OrchestratePipelineStages = false;
                training.MultiGpuStrategy = "none";
                training.GradientAccumulationSteps = 1;
                training.AutoDeviceMap = true;
                training.QuantizationProfile = "dynamic-int8";
                training.QuantizationCalibrationSamples = 64;
                training.EnableQatPath = false;
                training.QatFineTuneSteps = 100;
                training.FineTuningOrchestration = false;
                training.FineTuneStageSft = true;
                training.FineTuneStageDpo = false;
                training.FineTuneStageRlhf = false;
                training.RlhfFeedbackSource = "inline";
                training.RlhfFeedbackPath = string.Empty;
                training.RewardModelingEnabled = false;
                training.SafetyPolicyMode = "standard";
                training.ExportOnnx = false;
                training.ExportGguf = false;
                break;
            case "Serious":
                training.BatchSize = 12;
                training.MaxSteps = Math.Max(training.MaxSteps, 1800);
                training.LearningRate = 6e-5;
                training.EvalEvery = 40;
                training.Optimizer = "adamw";
                training.Scheduler = "cosine";
                training.WarmupSteps = 250;
                training.MixedPrecision = true;
                training.Precision = "fp16";
                training.EnableGradientClipping = true;
                training.GradientClipNorm = 1.0;
                training.CheckpointEvery = 150;
                training.EnableDeduplication = true;
                training.RemoveDuplicateLines = true;
                training.RemoveDuplicateParagraphs = true;
                training.NormalizeUnicode = true;
                training.CollapseWhitespace = true;
                training.CurriculumLearning = true;
                training.CurriculumWarmupRatio = 0.2;
                training.EvalSuite = "standard-10";
                training.DistributedTraining = false;
                training.OrchestratePipelineStages = false;
                training.MultiGpuStrategy = "none";
                training.GradientAccumulationSteps = 2;
                training.AutoDeviceMap = true;
                training.QuantizationProfile = "ptq-int8";
                training.QuantizationCalibrationSamples = 96;
                training.EnableQatPath = false;
                training.QatFineTuneSteps = 200;
                training.FineTuningOrchestration = false;
                training.FineTuneStageSft = true;
                training.FineTuneStageDpo = false;
                training.FineTuneStageRlhf = false;
                training.RlhfFeedbackSource = "jsonl-human-feedback";
                training.RlhfFeedbackPath = string.Empty;
                training.RewardModelingEnabled = true;
                training.SafetyPolicyMode = "standard";
                training.ExportOnnx = false;
                training.ExportGguf = false;
                break;
            case "Research":
                training.BatchSize = 16;
                training.MaxSteps = Math.Max(training.MaxSteps, 1500);
                training.LearningRate = 1.5e-4;
                training.EvalEvery = 40;
                training.Optimizer = "adafactor";
                training.Scheduler = "cosine";
                training.WarmupSteps = 600;
                training.MixedPrecision = true;
                training.Precision = "bf16";
                training.EnableGradientClipping = true;
                training.GradientClipNorm = 0.8;
                training.CheckpointEvery = 100;
                training.EnableDeduplication = true;
                training.RemoveDuplicateLines = true;
                training.RemoveDuplicateParagraphs = true;
                training.NormalizeUnicode = true;
                training.CollapseWhitespace = true;
                training.CurriculumLearning = true;
                training.CurriculumWarmupRatio = 0.3;
                training.DistributedTraining = false;
                training.OrchestratePipelineStages = false;
                training.MultiGpuStrategy = "none";
                training.GradientAccumulationSteps = 2;
                training.AutoDeviceMap = true;
                training.EnablePostTrainingQuantization = false;
                training.QuantizationProfile = "ptq-int8";
                training.QuantizationCalibrationSamples = 128;
                training.EnableQatPath = true;
                training.QatFineTuneSteps = 500;
                training.EvalSuite = "full-20";
                training.FineTuningOrchestration = true;
                training.FineTuneStageSft = true;
                training.FineTuneStageDpo = true;
                training.FineTuneStageRlhf = true;
                training.RlhfFeedbackSource = "jsonl-human-feedback";
                training.RlhfFeedbackPath = string.Empty;
                training.RewardModelingEnabled = true;
                training.SafetyPolicyMode = "research";
                training.ExportOnnx = false;
                training.ExportGguf = true;
                break;
            case "Cluster":
                training.Optimizer = "adamw";
                training.Scheduler = "cosine";
                training.WarmupSteps = 500;
                training.MixedPrecision = true;
                training.Precision = "bf16";
                training.EnableGradientClipping = true;
                training.GradientClipNorm = 1.0;
                training.CheckpointEvery = 100;
                training.EnableDeduplication = true;
                training.RemoveDuplicateLines = true;
                training.RemoveDuplicateParagraphs = true;
                training.NormalizeUnicode = true;
                training.CollapseWhitespace = true;
                training.CurriculumLearning = true;
                training.CurriculumWarmupRatio = 0.25;
                training.DistributedTraining = true;
                training.OrchestratePipelineStages = true;
                training.PipelineRunDataStage = true;
                training.PipelineRunPreprocessStage = true;
                training.PipelineRunTrainStage = true;
                training.PipelineRunEvalStage = true;
                training.MultiGpuStrategy = "ddp";
                training.GradientAccumulationSteps = 2;
                training.AutoDeviceMap = true;
                training.QuantizationProfile = "ptq-int4";
                training.QuantizationCalibrationSamples = 256;
                training.EnableQatPath = true;
                training.QatFineTuneSteps = 800;
                training.EvalSuite = "full-20";
                training.FineTuningOrchestration = true;
                training.FineTuneStageSft = true;
                training.FineTuneStageDpo = true;
                training.FineTuneStageRlhf = true;
                training.RlhfFeedbackSource = "external-import";
                training.RlhfFeedbackPath = string.Empty;
                training.RewardModelingEnabled = true;
                training.SafetyPolicyMode = "strict";
                training.ExportOnnx = false;
                training.ExportGguf = true;
                break;
        }
    }

    public static void ApplySamplingProfile(string profile, TokenizerKind tokenizerKind, SamplingConfig sampling)
    {
        // Chat-first, user-friendly defaults:
        // - random seed by default for natural behavior
        // - slightly lower temperature/top-k on fragile tokenizers
        // - profile-aware output budget
        profile = string.IsNullOrWhiteSpace(profile) ? "Balanced" : profile.Trim();

        sampling.Seed = -1;
        sampling.Greedy = false;

        switch (profile)
        {
            case "Tiny":
                sampling.Temperature = 0.40;
                sampling.TopK = 24;
                sampling.MaxNewTokens = 80;
                break;
            case "Serious":
                sampling.Temperature = 0.42;
                sampling.TopK = 28;
                sampling.MaxNewTokens = 120;
                break;
            case "Research":
                sampling.Temperature = 0.48;
                sampling.TopK = 40;
                sampling.MaxNewTokens = 160;
                break;
            case "Cluster":
                sampling.Temperature = 0.42;
                sampling.TopK = 30;
                sampling.MaxNewTokens = 140;
                break;
            case "Balanced":
            default:
                sampling.Temperature = 0.45;
                sampling.TopK = 30;
                sampling.MaxNewTokens = 100;
                break;
        }

        // Tokenizer-aware safety tuning.
        if (tokenizerKind is TokenizerKind.Character or TokenizerKind.HierarchicalExperimental)
        {
            sampling.Temperature = Math.Min(sampling.Temperature, 0.40);
            sampling.TopK = Math.Min(sampling.TopK, 24);
            sampling.MaxNewTokens = Math.Min(sampling.MaxNewTokens, 100);
        }
        else if (tokenizerKind is TokenizerKind.ByteLevelBpe or TokenizerKind.SimpleBpe or TokenizerKind.HybridFallback)
        {
            sampling.Temperature = Math.Min(sampling.Temperature, 0.45);
            sampling.TopK = Math.Min(sampling.TopK, 30);
        }
    }

    public static string DescribeTrainingProfile(string profile) => profile switch
    {
        "Tiny" => "Preset Tiny: rapido e leggero per test locali.",
        "Balanced" => "Preset Balanced: buon compromesso stabilita/velocita.",
        "Serious" => "Preset Serious: training robusto con dedup e curriculum.",
        "Research" => "Preset Research: setup piu spinto per esperimenti approfonditi.",
        "Cluster" => "Preset Cluster: configurazione orientata a distribuzione e benchmark completi.",
        _ => "Preset Custom: modifica manuale completa."
    };
}
