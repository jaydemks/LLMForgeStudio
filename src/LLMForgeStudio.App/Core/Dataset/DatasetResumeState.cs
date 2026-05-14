namespace LLMForgeStudio.App.Core.Dataset;

public sealed class DatasetResumeState
{
    public string ManifestPath { get; set; } = string.Empty;
    public int LastCompletedShardIndex { get; set; } = -1;
    public DateTimeOffset UpdatedAtUtc { get; set; } = DateTimeOffset.UtcNow;
}
