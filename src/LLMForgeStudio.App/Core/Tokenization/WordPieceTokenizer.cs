namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class WordPieceTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _stoi = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _itos = new();
    private const string Unknown = "[UNK]";

    public string Name => "WordPiece (Lite)";
    public TokenizerKind Kind => TokenizerKind.WordPiece;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        if (config.LowercaseBeforeTraining) text = text.ToLowerInvariant();

        var freq = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            [Unknown] = 1,
            ["▁"] = 1
        };

        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        foreach (var word in words)
        {
            var marked = "▁" + word;
            for (var i = 0; i < marked.Length; i++)
            {
                var maxLen = Math.Min(6, marked.Length - i);
                for (var len = 1; len <= maxLen; len++)
                {
                    var sub = marked.Substring(i, len);
                    var piece = i == 0 ? sub : "##" + sub;
                    freq[piece] = freq.GetValueOrDefault(piece) + 1;
                }
            }
        }

        var vocab = freq
            .Where(kv => kv.Value >= config.MinFrequency || kv.Key is Unknown or "▁")
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Take(Math.Max(64, config.TargetVocabSize))
            .Select(kv => (Token: kv.Key, Frequency: kv.Value))
            .ToList();

        if (!vocab.Any(v => v.Token == Unknown)) vocab.Insert(0, (Unknown, 1));

        _stoi.Clear();
        _itos.Clear();
        for (var i = 0; i < vocab.Count; i++)
        {
            _stoi[vocab[i].Token] = i;
            _itos[i] = vocab[i].Token;
        }

        Vocabulary = vocab.Select((v, i) => new VocabularyItem(i, v.Token, v.Frequency)).ToList();
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            var marked = "▁" + word;
            var cursor = 0;
            while (cursor < marked.Length)
            {
                var matched = false;
                var maxLen = Math.Min(6, marked.Length - cursor);
                for (var len = maxLen; len >= 1; len--)
                {
                    var sub = marked.Substring(cursor, len);
                    var token = cursor == 0 ? sub : "##" + sub;
                    if (_stoi.TryGetValue(token, out var id))
                    {
                        ids.Add(id);
                        cursor += len;
                        matched = true;
                        break;
                    }
                }

                if (!matched)
                {
                    ids.Add(0);
                    cursor++;
                }
            }
        }

        return ids;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var sb = new System.Text.StringBuilder();
        foreach (var id in ids)
        {
            if (!_itos.TryGetValue(id, out var token) || token == Unknown) continue;
            if (token.StartsWith("##", StringComparison.Ordinal))
                sb.Append(token.AsSpan(2));
            else
                sb.Append(token);
        }

        return sb.ToString().Replace("▁", " ").Trim();
    }
}
