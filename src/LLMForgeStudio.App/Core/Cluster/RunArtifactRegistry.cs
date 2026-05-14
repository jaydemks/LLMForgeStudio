using System.Text.Json;

namespace LLMForgeStudio.App.Core.Cluster;

public static class RunArtifactRegistry
{
    public static async Task<string> UpdateAsync(string runDirectory)
    {
        Directory.CreateDirectory(runDirectory);
        var record = BuildRecord(runDirectory);
        var path = Path.Combine(runDirectory, "artifact_registry.json");
        var json = JsonSerializer.Serialize(record, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, json);
        return path;
    }

    public static RunArtifactRecord BuildRecord(string runDirectory)
    {
        var fullRun = Path.GetFullPath(runDirectory);

        var checkpoints = Directory.Exists(fullRun)
            ? Directory.GetFiles(fullRun, "checkpoint_step_*.pt", SearchOption.TopDirectoryOnly).OrderBy(p => p).ToList()
            : new List<string>();

        return new RunArtifactRecord
        {
            RunDirectory = fullRun,
            IndexedAtUtc = DateTimeOffset.UtcNow,
            ManifestPath = Path.Combine(fullRun, "checkpoint_manifest.json"),
            ClusterStatePath = Path.Combine(fullRun, "cluster_run_state.json"),
            ClusterHeartbeatPath = Path.Combine(fullRun, "cluster_heartbeat.json"),
            PipelineStageStatePath = Path.Combine(fullRun, "pipeline_stage_state.json"),
            EvalSummaryPath = Path.Combine(fullRun, "eval_summary.json"),
            EvalScorecardPath = Path.Combine(fullRun, "eval_scorecard.md"),
            EvalTrendPath = Path.Combine(fullRun, "eval_trend.json"),
            EvalRegressionPath = Path.Combine(fullRun, "eval_regression.json"),
            EvalReleaseScorecardPath = Path.Combine(fullRun, "release_candidate_scorecard.md"),
            CheckpointPaths = checkpoints
        };
    }
}
