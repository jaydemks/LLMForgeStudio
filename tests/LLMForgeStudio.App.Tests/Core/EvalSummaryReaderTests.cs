using LLMForgeStudio.App.Core.Training;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class EvalSummaryReaderTests
{
    [Fact]
    public async Task TryReadAsync_ParsesSummary()
    {
        var root = Path.Combine(Path.GetTempPath(), $"llmforge-eval-summary-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        var path = Path.Combine(root, "eval_summary.json");

        await File.WriteAllTextAsync(path, "{\"eval_suite\":\"full-20\",\"num_benchmarks\":20,\"average_score\":77.5,\"band\":\"good\",\"release_gate_passed\":true,\"release_gate_threshold\":70.0}");

        try
        {
            var snap = await EvalSummaryReader.TryReadAsync(root);
            Assert.NotNull(snap);
            Assert.Equal("full-20", snap!.EvalSuite);
            Assert.Equal(20, snap.NumBenchmarks);
            Assert.Equal(77.5, snap.AverageScore, 3);
            Assert.True(snap.ReleaseGatePassed);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, recursive: true);
        }
    }
}
