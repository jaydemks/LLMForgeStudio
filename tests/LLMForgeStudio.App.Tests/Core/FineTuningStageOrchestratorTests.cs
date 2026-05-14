using LLMForgeStudio.App.Core.Alignment;
using LLMForgeStudio.App.Core.Training;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class FineTuningStageOrchestratorTests
{
    [Fact]
    public async Task PrepareAsync_WritesStageArtifact_WhenOrchestrationEnabled()
    {
        var runDir = Path.Combine(Path.GetTempPath(), $"llmforge-ft-{Guid.NewGuid():N}");
        Directory.CreateDirectory(runDir);
        try
        {
            var cfg = new TrainingConfig
            {
                FineTuningOrchestration = true,
                FineTuneStageSft = true,
                FineTuneStageDpo = false,
                FineTuneStageRlhf = true,
                RlhfFeedbackSource = "external-import"
            };
            var feedbackPath = Path.Combine(runDir, "feedback.jsonl");
            await File.WriteAllTextAsync(feedbackPath, "{\"prompt\":\"p\",\"chosen\":\"c\",\"rejected\":\"r\"}\n");
            cfg.RlhfFeedbackPath = feedbackPath;

            var outPath = await FineTuningStageOrchestrator.PrepareAsync("{\"prompt\":\"a\",\"response\":\"b\"}", cfg, runDir);
            Assert.True(File.Exists(outPath));
            Assert.True(File.Exists(Path.Combine(runDir, "fine_tuning_stages.json")));
            Assert.True(File.Exists(Path.Combine(runDir, "rlhf_feedback_import.json")));
        }
        finally
        {
            if (Directory.Exists(runDir)) Directory.Delete(runDir, recursive: true);
        }
    }
}
