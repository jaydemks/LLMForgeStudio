namespace LLMForgeStudio.App.Core.Dataset;

public sealed class TextCleanerConfig
{
    public bool NormalizeNewLines { get; set; } = true;
    public bool TrimLines { get; set; } = true;
    public bool RemoveEmptyLines { get; set; } = false;
    public bool Lowercase { get; set; } = false;
}
