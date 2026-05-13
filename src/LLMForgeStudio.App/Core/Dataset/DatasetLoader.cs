namespace LLMForgeStudio.App.Core.Dataset;

public static class DatasetLoader
{
    public static async Task<string> LoadTextAsync(string path, TextCleanerConfig config)
    {
        var raw = await File.ReadAllTextAsync(path);
        return TextCleaner.Clean(raw, config);
    }
}
