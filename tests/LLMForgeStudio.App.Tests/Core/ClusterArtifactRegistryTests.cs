using LLMForgeStudio.App.Core.Cluster;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class ClusterArtifactRegistryTests
{
    [Fact]
    public async Task UpdateAsync_WritesArtifactRegistryFile()
    {
        var root = Path.Combine(Path.GetTempPath(), $"llmforge-artifacts-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);

        await File.WriteAllTextAsync(Path.Combine(root, "checkpoint_manifest.json"), "{}");
        await File.WriteAllTextAsync(Path.Combine(root, "eval_summary.json"), "{}");
        await File.WriteAllTextAsync(Path.Combine(root, "eval_regression.json"), "{}");
        await File.WriteAllTextAsync(Path.Combine(root, "release_candidate_scorecard.md"), "ok");
        await File.WriteAllTextAsync(Path.Combine(root, "cluster_run_state.json"), "{}");
        await File.WriteAllTextAsync(Path.Combine(root, "checkpoint_step_10.pt"), "bin");

        try
        {
            var path = await RunArtifactRegistry.UpdateAsync(root);
            Assert.True(File.Exists(path));

            var json = await File.ReadAllTextAsync(path);
            Assert.Contains("checkpoint_manifest.json", json);
            Assert.Contains("checkpoint_step_10.pt", json);
            Assert.Contains("eval_summary.json", json);
            Assert.Contains("eval_regression.json", json);
            Assert.Contains("release_candidate_scorecard.md", json);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
