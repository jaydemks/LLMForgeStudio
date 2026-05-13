using System.Text.Json;

namespace LLMForgeStudio.App.Core.Project;

public static class ProjectStore
{
    public static async Task SaveAsync(ForgeProject project, string path)
    {
        var json = JsonSerializer.Serialize(project, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(path, json);
    }

    public static async Task<ForgeProject> LoadAsync(string path)
    {
        var json = await File.ReadAllTextAsync(path);
        return JsonSerializer.Deserialize<ForgeProject>(json) ?? new ForgeProject();
    }
}
