using System.Text.Json;

namespace LLMForgeStudio.App.Core.Cluster;

public static class ClusterProfileManager
{
    public static IReadOnlyList<ClusterProfile> BuiltIns =>
    [
        new() { Name = "single-node", Orchestrator = "local", WorldSize = 1, MaxRetries = 0, AutoResume = true },
        new() { Name = "workstation-multigpu", Orchestrator = "local", WorldSize = 2, MaxRetries = 1, AutoResume = true },
        new() { Name = "cluster-standard", Orchestrator = "scheduler", WorldSize = 8, MaxRetries = 2, AutoResume = true },
        new() { Name = "cluster-sharedfs", Orchestrator = "sharedfs", WorldSize = 8, MaxRetries = 2, AutoResume = true }
    ];

    public static async Task<string> SaveProfilesAsync(IEnumerable<ClusterProfile> profiles, string directory)
    {
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "cluster_profiles.json");
        var json = JsonSerializer.Serialize(profiles, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, json);
        return path;
    }

    public static async Task<IReadOnlyList<ClusterProfile>> LoadProfilesAsync(string path)
    {
        if (!File.Exists(path)) return BuiltIns;
        var json = await File.ReadAllTextAsync(path);
        return JsonSerializer.Deserialize<List<ClusterProfile>>(json) ?? BuiltIns;
    }

    public static ClusterProfile Resolve(string name)
    {
        return BuiltIns.FirstOrDefault(p => string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase))
            ?? BuiltIns[0];
    }
}
