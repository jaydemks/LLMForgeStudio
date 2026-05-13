using LLMForgeStudio.App.Core.Generation;
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
            Assert.Equal(project.Sampling.TopK, loaded.Sampling.TopK);
            Assert.Equal(project.SelectedSection, loaded.SelectedSection);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
