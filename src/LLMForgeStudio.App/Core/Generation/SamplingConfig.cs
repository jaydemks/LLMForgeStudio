namespace LLMForgeStudio.App.Core.Generation;

public sealed class SamplingConfig
{
    public double Temperature { get; set; } = 0.8;
    public int TopK { get; set; } = 40;
    public int Seed { get; set; } = 42;
    public bool Greedy { get; set; } = false;
    public int MaxNewTokens { get; set; } = 200;
}
