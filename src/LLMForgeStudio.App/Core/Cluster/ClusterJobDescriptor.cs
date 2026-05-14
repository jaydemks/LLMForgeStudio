using System.Text.Json;
using LLMForgeStudio.App.Core.Backend;

namespace LLMForgeStudio.App.Core.Cluster;

public sealed class ClusterJobDescriptor
{
    public string ProfileName { get; set; } = "single-node";
    public string Orchestrator { get; set; } = "local";
    public int WorldSize { get; set; } = 1;
    public int MaxRetries { get; set; } = 0;
    public string MultiGpuStrategy { get; set; } = "none";
    public int GradientAccumulationSteps { get; set; } = 1;
    public string JobSpecPath { get; set; } = string.Empty;

    public static async Task<string> WriteAsync(string outputDirectory, ClusterJobDescriptor descriptor)
    {
        Directory.CreateDirectory(outputDirectory);
        var path = Path.Combine(outputDirectory, "cluster_job_descriptor.json");
        var json = JsonSerializer.Serialize(descriptor, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, json);
        return path;
    }

    public static ClusterJobDescriptor FromSpec(BackendJobSpec spec)
    {
        return new ClusterJobDescriptor
        {
            ProfileName = spec.ClusterProfileName,
            Orchestrator = spec.Training.ClusterOrchestrator,
            WorldSize = Math.Max(1, spec.Training.ClusterWorldSize),
            MaxRetries = Math.Max(0, spec.Training.ClusterMaxRetries),
            MultiGpuStrategy = string.IsNullOrWhiteSpace(spec.Training.MultiGpuStrategy) ? "none" : spec.Training.MultiGpuStrategy,
            GradientAccumulationSteps = Math.Max(1, spec.Training.GradientAccumulationSteps)
        };
    }
}
