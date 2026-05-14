namespace LLMForgeStudio.App.Core.Dataset;

public sealed class DatasetShardItem
{
    public string RelativePath { get; set; } = string.Empty;
    public long Bytes { get; set; }
    public long Characters { get; set; }
    public string Sha256 { get; set; } = string.Empty;
}
