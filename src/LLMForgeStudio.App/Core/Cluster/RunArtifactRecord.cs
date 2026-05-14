namespace LLMForgeStudio.App.Core.Cluster;

public sealed class RunArtifactRecord
{
    public string RunDirectory { get; set; } = string.Empty;
    public DateTimeOffset IndexedAtUtc { get; set; } = DateTimeOffset.UtcNow;
    public string ManifestPath { get; set; } = string.Empty;
    public string ClusterStatePath { get; set; } = string.Empty;
    public string ClusterHeartbeatPath { get; set; } = string.Empty;
    public string PipelineStageStatePath { get; set; } = string.Empty;
    public string EvalSummaryPath { get; set; } = string.Empty;
    public string EvalScorecardPath { get; set; } = string.Empty;
    public string EvalTrendPath { get; set; } = string.Empty;
    public string EvalRegressionPath { get; set; } = string.Empty;
    public string EvalReleaseScorecardPath { get; set; } = string.Empty;
    public List<string> CheckpointPaths { get; set; } = new();
}
