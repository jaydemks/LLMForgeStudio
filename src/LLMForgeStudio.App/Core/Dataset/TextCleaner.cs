using System.Text.RegularExpressions;

namespace LLMForgeStudio.App.Core.Dataset;

public static class TextCleaner
{
    public static string Clean(string text, TextCleanerConfig config)
    {
        if (string.IsNullOrEmpty(text)) return string.Empty;

        if (config.NormalizeNewLines)
            text = text.Replace("\r\n", "\n").Replace("\r", "\n");

        if (config.Lowercase)
            text = text.ToLowerInvariant();

        if (config.TrimLines || config.RemoveEmptyLines)
        {
            var lines = text.Split('\n')
                .Select(line => config.TrimLines ? line.Trim() : line);

            if (config.RemoveEmptyLines)
                lines = lines.Where(line => !string.IsNullOrWhiteSpace(line));

            text = string.Join('\n', lines);
        }

        return text;
    }

    public static DatasetStats Analyze(string text, int previewLength = 1800)
    {
        var lines = text.Length == 0 ? 0 : text.Split('\n').Length;
        var words = Regex.Matches(text, @"\S+").Count;
        var uniqueChars = text.Distinct().Count();
        var preview = text.Length <= previewLength ? text : text[..previewLength] + "\n...";
        return new DatasetStats(text.Length, lines, words, uniqueChars, preview);
    }
}
