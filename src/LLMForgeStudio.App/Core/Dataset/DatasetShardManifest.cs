namespace LLMForgeStudio.App.Core.Dataset;

public sealed class DatasetShardManifest
{
    public string FormatVersion { get; set; } = "0.1";
    public string RootDirectory { get; set; } = string.Empty;
    public List<string> Shards { get; set; } = new();
    public List<DatasetShardItem> ShardItems { get; set; } = new();
    public long TotalBytes { get; set; }
    public long TotalCharacters { get; set; }
}
