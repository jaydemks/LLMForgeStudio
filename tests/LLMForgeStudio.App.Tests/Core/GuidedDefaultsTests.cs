using LLMForgeStudio.App.Core.Guidance;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class GuidedDefaultsTests
{
    [Fact]
    public void TrainingProfile_Tiny_AppliesExpectedCoreValues()
    {
        var cfg = new TrainingConfig();
        GuidedDefaultsEngine.ApplyTrainingProfile("Tiny", cfg);

        Assert.Equal(8, cfg.BatchSize);
        Assert.Equal("adamw", cfg.Optimizer);
        Assert.Equal("none", cfg.Scheduler);
        Assert.False(cfg.DistributedTraining);
        Assert.Equal("basic", cfg.EvalSuite);
    }

    [Fact]
    public void TrainingProfile_Cluster_EnablesDistributedAndFullEval()
    {
        var cfg = new TrainingConfig();
        GuidedDefaultsEngine.ApplyTrainingProfile("Cluster", cfg);

        Assert.True(cfg.DistributedTraining);
        Assert.True(cfg.OrchestratePipelineStages);
        Assert.True(cfg.PipelineRunTrainStage);
        Assert.Equal("full-20", cfg.EvalSuite);
        Assert.True(cfg.EnableDeduplication);
        Assert.True(cfg.CurriculumLearning);
        Assert.Equal("ddp", cfg.MultiGpuStrategy);
        Assert.Equal(2, cfg.GradientAccumulationSteps);
        Assert.True(cfg.RewardModelingEnabled);
        Assert.Equal("strict", cfg.SafetyPolicyMode);
        Assert.True(cfg.ExportOnnx);
        Assert.True(cfg.ExportGguf);
        Assert.True(cfg.EnableQatPath);
        Assert.True(cfg.QatFineTuneSteps >= 1);
    }

    [Fact]
    public void TokenizerPreset_Character_SetsDeterministicValues()
    {
        var tokenizer = new TokenizerConfig();
        GuidedDefaultsEngine.ApplyTokenizerPreset(TokenizerKind.Character, "hello world", tokenizer);

        Assert.Equal(256, tokenizer.TargetVocabSize);
        Assert.Equal(1, tokenizer.MinFrequency);
        Assert.Equal(0, tokenizer.MaxMerges);
    }

    [Fact]
    public void TokenizerPreset_ByteLevelBpe_EnablesExpectedProductionDefaults()
    {
        var tokenizer = new TokenizerConfig();
        GuidedDefaultsEngine.ApplyTokenizerPreset(TokenizerKind.ByteLevelBpe, "a b c d e f g h", tokenizer);

        Assert.Equal(2000, tokenizer.TargetVocabSize);
        Assert.Equal(2, tokenizer.MinFrequency);
        Assert.Equal(1000, tokenizer.MaxMerges);
        Assert.True(tokenizer.KeepPunctuationAsTokens);
    }
}
