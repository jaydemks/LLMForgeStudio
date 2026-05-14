using System.Text;
using System.Text.Json;

namespace LLMForgeStudio.App.Core.Alignment;

public static class RlhfFeedbackCollector
{
    public static async Task<string> SaveJsonlAsync(IEnumerable<RlhfFeedbackRecord> records, string outputPath)
    {
        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        var sb = new StringBuilder();
        foreach (var r in records)
        {
            if (string.IsNullOrWhiteSpace(r.Prompt) || string.IsNullOrWhiteSpace(r.Chosen))
                continue;

            var payload = new
            {
                prompt = r.Prompt,
                chosen = r.Chosen,
                rejected = r.Rejected ?? string.Empty
            };
            sb.AppendLine(JsonSerializer.Serialize(payload));
        }

        await File.WriteAllTextAsync(outputPath, sb.ToString(), Encoding.UTF8);
        return outputPath;
    }
}
