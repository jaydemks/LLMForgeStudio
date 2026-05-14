using System.Text.Json;

namespace LLMForgeStudio.App.Core.Training;

public static class EvalSummaryReader
{
    public static async Task<EvalSummarySnapshot?> TryReadAsync(string runDirectory)
    {
        var path = Path.Combine(runDirectory, "eval_summary.json");
        if (!File.Exists(path)) return null;

        try
        {
            var json = await File.ReadAllTextAsync(path);
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            return new EvalSummarySnapshot
            {
                EvalSuite = root.TryGetProperty("eval_suite", out var s) ? (s.GetString() ?? "basic") : "basic",
                NumBenchmarks = root.TryGetProperty("num_benchmarks", out var n) ? n.GetInt32() : 0,
                AverageScore = root.TryGetProperty("average_score", out var a) ? a.GetDouble() : 0,
                Band = root.TryGetProperty("band", out var b) ? (b.GetString() ?? "unknown") : "unknown",
                ReleaseGatePassed = root.TryGetProperty("release_gate_passed", out var p) && p.GetBoolean(),
                ReleaseGateThreshold = root.TryGetProperty("release_gate_threshold", out var t) ? t.GetDouble() : 0,
            };
        }
        catch
        {
            return null;
        }
    }
}
