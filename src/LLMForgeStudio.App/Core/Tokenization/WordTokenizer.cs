using System.Text;
using System.Text.RegularExpressions;

namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class WordTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _stoi = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _itos = new();
    private const string Unknown = "<unk>";

    public string Name => "Word-level";
    public TokenizerKind Kind => TokenizerKind.Word;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        if (config.LowercaseBeforeTraining) text = text.ToLowerInvariant();
        var tokens = Split(text, config.KeepPunctuationAsTokens).ToList();
        var freq = tokens.GroupBy(t => t).ToDictionary(g => g.Key, g => g.Count());
        var ordered = freq.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).Select(kv => kv.Key).ToList();
        ordered.Insert(0, Unknown);

        _stoi.Clear();
        _itos.Clear();
        for (var i = 0; i < ordered.Count; i++)
        {
            _stoi[ordered[i]] = i;
            _itos[i] = ordered[i];
        }
        Vocabulary = ordered.Select((token, id) => new VocabularyItem(id, token, freq.GetValueOrDefault(token))).ToList();
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (var token in Split(text, true))
            ids.Add(_stoi.TryGetValue(token, out var id) ? id : 0);
        return ids;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var tokens = ids.Select(id => _itos.TryGetValue(id, out var token) ? token : Unknown).ToList();
        var sb = new StringBuilder();

        for (var i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            if (token == "\n")
            {
                sb.Append('\n');
                continue;
            }

            var isPunctuation = Regex.IsMatch(token, @"^[^\w\s]$");
            if (sb.Length > 0 && !isPunctuation && sb[^1] != '\n') sb.Append(' ');
            sb.Append(token);
        }

        return sb.ToString();
    }

    private static IEnumerable<string> Split(string text, bool keepPunctuation)
    {
        var pattern = keepPunctuation ? @"\w+|[^\w\s]|\n" : @"\w+";
        return Regex.Matches(text, pattern).Select(m => m.Value);
    }
}
