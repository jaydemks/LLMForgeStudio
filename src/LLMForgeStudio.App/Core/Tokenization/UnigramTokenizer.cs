namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class UnigramTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _stoi = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _itos = new();
    private const string Unknown = "<unk>";

    public string Name => "Unigram (Lite)";
    public TokenizerKind Kind => TokenizerKind.Unigram;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        if (config.LowercaseBeforeTraining) text = text.ToLowerInvariant();

        var counts = new Dictionary<string, int>(StringComparer.Ordinal)
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
                var maxLen = Math.Min(8, marked.Length - i);
                for (var len = 1; len <= maxLen; len++)
                {
                    var piece = marked.Substring(i, len);
                    counts[piece] = counts.GetValueOrDefault(piece) + 1;
                }
            }
        }

        var vocab = counts
            .Where(kv => kv.Value >= config.MinFrequency || kv.Key is Unknown or "▁")
            .OrderByDescending(kv => kv.Value)
            .ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Take(Math.Max(64, config.TargetVocabSize))
            .Select(kv => (Token: kv.Key, Frequency: kv.Value))
            .ToList();

        if (!vocab.Any(v => v.Token == Unknown)) vocab.Insert(0, (Unknown, 1));
        if (!vocab.Any(v => v.Token == "▁")) vocab.Insert(1, ("▁", 1));

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
            var idx = 0;
            while (idx < marked.Length)
            {
                var matched = false;
                var maxLen = Math.Min(8, marked.Length - idx);
                for (var len = maxLen; len >= 1; len--)
                {
                    var piece = marked.Substring(idx, len);
                    if (_stoi.TryGetValue(piece, out var id))
                    {
                        ids.Add(id);
                        idx += len;
                        matched = true;
                        break;
                    }
                }

                if (!matched)
                {
                    ids.Add(0);
                    idx++;
                }
            }
        }

        return ids;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var pieces = ids.Select(id => _itos.TryGetValue(id, out var token) ? token : Unknown)
            .Where(t => t != Unknown)
            .ToArray();

        return string.Concat(pieces).Replace("▁", " ").Trim();
    }
}
