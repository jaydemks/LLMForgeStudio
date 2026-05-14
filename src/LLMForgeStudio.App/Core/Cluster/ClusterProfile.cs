namespace LLMForgeStudio.App.Core.Cluster;

public sealed class ClusterProfile
{
    public string Name { get; set; } = "single-node";
    public string Orchestrator { get; set; } = "local";
    public int WorldSize { get; set; } = 1;
    public int MaxRetries { get; set; } = 0;
    public bool AutoResume { get; set; } = true;
}
