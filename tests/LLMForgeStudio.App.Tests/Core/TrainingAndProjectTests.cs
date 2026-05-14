using LLMForgeStudio.App.Core.Generation;
using LLMForgeStudio.App.Core.Cluster;
using LLMForgeStudio.App.Core.Project;
using LLMForgeStudio.App.Core.Tokenization;
using LLMForgeStudio.App.Core.Training;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class TrainingAndProjectTests
{
    [Fact]
    public void BatchPreview_HasShiftedTargets()
    {
        var tokens = Enumerable.Range(1, 400).ToList();
        var preview = TrainingBatchBuilder.BuildPreview(tokens, 16);

        for (var i = 0; i < preview.X.Count - 1; i++)
            Assert.Equal(preview.X[i + 1], preview.Y[i]);
    }

    [Fact]
    public void Sampler_TopK1_EqualsGreedy()
    {
        var logits = new double[] { 0.1, 0.3, 2.2, 0.9 };
        var config = new SamplingConfig { Greedy = false, TopK = 1, Temperature = 0.8, Seed = 42 };
        var id = Sampler.Sample(logits, config);
        Assert.Equal(2, id);
    }

    [Fact]
    public async Task ProjectStore_SaveLoad_RoundTrip()
    {
        var project = new ForgeProject
        {
            DatasetPath = "a.txt",
            DatasetText = "hello",
            Tokenizer = new TokenizerConfig { TargetVocabSize = 123 },
            Model = new ModelConfig { BlockSize = 64 },
            Training = new TrainingConfig { MaxSteps = 50 },
            Sampling = new SamplingConfig { TopK = 10 },
            SelectedSection = "Training"
        };

        var path = Path.Combine(Path.GetTempPath(), $"llmforge-test-{Guid.NewGuid():N}.json");
        try
        {
            await ProjectStore.SaveAsync(project, path);
            var loaded = await ProjectStore.LoadAsync(path);

            Assert.Equal(project.DatasetPath, loaded.DatasetPath);
            Assert.Equal(project.DatasetText, loaded.DatasetText);
            Assert.Equal(project.Tokenizer.TargetVocabSize, loaded.Tokenizer.TargetVocabSize);
            Assert.Equal(project.Model.BlockSize, loaded.Model.BlockSize);
            Assert.Equal(project.Training.MaxSteps, loaded.Training.MaxSteps);
            Assert.Equal(project.Training.Optimizer, loaded.Training.Optimizer);
            Assert.Equal(project.Sampling.TopK, loaded.Sampling.TopK);
            Assert.Equal(project.SelectedSection, loaded.SelectedSection);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task ProjectStore_SaveLoad_RoundTrip_AdvancedTrainingFields()
    {
        var project = new ForgeProject
        {
            Training = new TrainingConfig
            {
                BatchSize = 12,
                MaxSteps = 1234,
                LearningRate = 2e-4,
                EvalEvery = 25,
                Optimizer = "lion",
                Scheduler = "cosine",
                WarmupSteps = 100,
                MixedPrecision = true,
                Precision = "bf16",
                EnableGradientClipping = true,
                GradientClipNorm = 0.8,
                CheckpointEvery = 200,
                EnablePostTrainingQuantization = true,
                QuantizationProfile = "ptq-int8",
                QuantizationCalibrationSamples = 128,
                EnableQatPath = true,
                QatFineTuneSteps = 333,
                EnableDeduplication = true,
                RemoveDuplicateLines = true,
                RemoveDuplicateParagraphs = true,
                NormalizeUnicode = true,
                CollapseWhitespace = true,
                CurriculumLearning = true,
                CurriculumWarmupRatio = 0.3,
                ResumeDatasetFromState = true,
                DeterministicShardShuffle = true,
                DataShuffleSeed = 1337,
                ClusterProfileName = "cluster-standard",
                ClusterOrchestrator = "scheduler",
                ClusterWorldSize = 8,
                ClusterMaxRetries = 3,
                ClusterHeartbeatSeconds = 3,
                OrchestratePipelineStages = true,
                PipelineRunDataStage = true,
                PipelineRunPreprocessStage = true,
                PipelineRunTrainStage = true,
                PipelineRunEvalStage = true,
                DistributedTraining = true,
                MultiGpuStrategy = "ddp",
                GradientAccumulationSteps = 4,
                AutoDeviceMap = true,
                AlignmentMode = "dpo",
                FineTuningOrchestration = true,
                FineTuneStageSft = true,
                FineTuneStageDpo = true,
                FineTuneStageRlhf = false,
                RlhfFeedbackSource = "jsonl-human-feedback",
                RlhfFeedbackPath = @"C:\tmp\rlhf_feedback.jsonl",
                RewardModelingEnabled = true,
                SafetyPolicyMode = "research",
                ExportOnnx = true,
                ExportGguf = true,
                EvalSuite = "full-20",
                ForceCpu = false
            }
        };

        var path = Path.Combine(Path.GetTempPath(), $"llmforge-test-advanced-{Guid.NewGuid():N}.json");
        try
        {
            await ProjectStore.SaveAsync(project, path);
            var loaded = await ProjectStore.LoadAsync(path);

            Assert.Equal("lion", loaded.Training.Optimizer);
            Assert.Equal("cosine", loaded.Training.Scheduler);
            Assert.Equal(100, loaded.Training.WarmupSteps);
            Assert.True(loaded.Training.MixedPrecision);
            Assert.Equal("bf16", loaded.Training.Precision);
            Assert.Equal(0.8, loaded.Training.GradientClipNorm, 3);
            Assert.Equal(200, loaded.Training.CheckpointEvery);
            Assert.True(loaded.Training.EnablePostTrainingQuantization);
            Assert.Equal("ptq-int8", loaded.Training.QuantizationProfile);
            Assert.Equal(128, loaded.Training.QuantizationCalibrationSamples);
            Assert.True(loaded.Training.EnableQatPath);
            Assert.Equal(333, loaded.Training.QatFineTuneSteps);
            Assert.True(loaded.Training.EnableDeduplication);
            Assert.True(loaded.Training.RemoveDuplicateLines);
            Assert.True(loaded.Training.RemoveDuplicateParagraphs);
            Assert.True(loaded.Training.CollapseWhitespace);
            Assert.True(loaded.Training.CurriculumLearning);
            Assert.Equal(0.3, loaded.Training.CurriculumWarmupRatio, 3);
            Assert.True(loaded.Training.ResumeDatasetFromState);
            Assert.True(loaded.Training.DeterministicShardShuffle);
            Assert.Equal(1337, loaded.Training.DataShuffleSeed);
            Assert.Equal("cluster-standard", loaded.Training.ClusterProfileName);
            Assert.Equal("scheduler", loaded.Training.ClusterOrchestrator);
            Assert.Equal(8, loaded.Training.ClusterWorldSize);
            Assert.Equal(3, loaded.Training.ClusterMaxRetries);
            Assert.Equal(3, loaded.Training.ClusterHeartbeatSeconds);
            Assert.True(loaded.Training.OrchestratePipelineStages);
            Assert.True(loaded.Training.PipelineRunDataStage);
            Assert.True(loaded.Training.PipelineRunPreprocessStage);
            Assert.True(loaded.Training.PipelineRunTrainStage);
            Assert.True(loaded.Training.PipelineRunEvalStage);
            Assert.True(loaded.Training.DistributedTraining);
            Assert.Equal("ddp", loaded.Training.MultiGpuStrategy);
            Assert.Equal(4, loaded.Training.GradientAccumulationSteps);
            Assert.True(loaded.Training.AutoDeviceMap);
            Assert.Equal("dpo", loaded.Training.AlignmentMode);
            Assert.True(loaded.Training.FineTuningOrchestration);
            Assert.True(loaded.Training.FineTuneStageSft);
            Assert.True(loaded.Training.FineTuneStageDpo);
            Assert.False(loaded.Training.FineTuneStageRlhf);
            Assert.Equal("jsonl-human-feedback", loaded.Training.RlhfFeedbackSource);
            Assert.Equal(@"C:\tmp\rlhf_feedback.jsonl", loaded.Training.RlhfFeedbackPath);
            Assert.True(loaded.Training.RewardModelingEnabled);
            Assert.Equal("research", loaded.Training.SafetyPolicyMode);
            Assert.True(loaded.Training.ExportOnnx);
            Assert.True(loaded.Training.ExportGguf);
            Assert.Equal("full-20", loaded.Training.EvalSuite);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task ClusterProfileManager_SaveLoad_RoundTrip()
    {
        var tmp = Path.Combine(Path.GetTempPath(), $"llmforge-cluster-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(tmp);
            var profiles = new[]
            {
                new ClusterProfile { Name = "a", Orchestrator = "local", WorldSize = 2, MaxRetries = 1, AutoResume = true },
                new ClusterProfile { Name = "b", Orchestrator = "scheduler", WorldSize = 8, MaxRetries = 2, AutoResume = false }
            };

            var path = await ClusterProfileManager.SaveProfilesAsync(profiles, tmp);
            var loaded = await ClusterProfileManager.LoadProfilesAsync(path);

            Assert.Equal(2, loaded.Count);
            Assert.Equal("b", loaded[1].Name);
            Assert.Equal(8, loaded[1].WorldSize);
        }
        finally
        {
            if (Directory.Exists(tmp)) Directory.Delete(tmp, recursive: true);
        }
    }
}
