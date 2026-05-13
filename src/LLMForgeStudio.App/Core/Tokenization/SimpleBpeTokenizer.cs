namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class SimpleBpeTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _stoi = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _itos = new();
    private readonly List<(string Left, string Right)> _merges = new();
    private const string Unknown = "<unk>";

    public string Name => "Simple BPE";
    public TokenizerKind Kind => TokenizerKind.SimpleBpe;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        if (config.LowercaseBeforeTraining) text = text.ToLowerInvariant();
        _merges.Clear();

        // BPE didattico: parte da caratteri e cerca coppie frequenti da fondere.
        // Non è ancora ottimizzato come HuggingFace Tokenizers/SentencePiece.
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Select(w => w.Select(c => c.ToString()).Concat(new[] { "</w>" }).ToList())
            .ToList();

        for (var step = 0; step < config.MaxMerges; step++)
        {
            var pairCounts = new Dictionary<(string, string), int>();
            foreach (var word in words)
            {
                for (var i = 0; i < word.Count - 1; i++)
                {
                    var pair = (word[i], word[i + 1]);
                    pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + 1;
                }
            }

            var best = pairCounts
                .Where(kv => kv.Value >= config.MinFrequency)
                .OrderByDescending(kv => kv.Value)
                .FirstOrDefault();

            if (best.Key == default) break;
            _merges.Add(best.Key);
            var merged = best.Key.Item1 + best.Key.Item2;

            foreach (var word in words)
            {
                for (var i = 0; i < word.Count - 1; i++)
                {
                    if (word[i] == best.Key.Item1 && word[i + 1] == best.Key.Item2)
                    {
                        word[i] = merged;
                        word.RemoveAt(i + 1);
                    }
                }
            }

            var currentVocab = words.SelectMany(w => w).Distinct().Count();
            if (currentVocab >= config.TargetVocabSize) break;
        }

        var pieces = words.SelectMany(w => w).GroupBy(p => p)
            .OrderByDescending(g => g.Count()).ThenBy(g => g.Key)
            .Select(g => (Token: g.Key, Frequency: g.Count()))
            .ToList();

        pieces.Insert(0, (Unknown, 0));
        _stoi.Clear();
        _itos.Clear();
        for (var i = 0; i < pieces.Count; i++)
        {
            _stoi[pieces[i].Token] = i;
            _itos[i] = pieces[i].Token;
        }

        Vocabulary = pieces.Select((p, id) => new VocabularyItem(id, p.Token, p.Frequency)).ToList();
    }

    public IReadOnlyList<int> Encode(string text)
    {
        var result = new List<int>();
        foreach (var rawWord in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            var pieces = rawWord.Select(c => c.ToString()).Concat(new[] { "</w>" }).ToList();
            foreach (var merge in _merges)
            {
                var merged = merge.Left + merge.Right;
                for (var i = 0; i < pieces.Count - 1; i++)
                {
                    if (pieces[i] == merge.Left && pieces[i + 1] == merge.Right)
                    {
                        pieces[i] = merged;
                        pieces.RemoveAt(i + 1);
                    }
                }
            }
            result.AddRange(pieces.Select(p => _stoi.TryGetValue(p, out var id) ? id : 0));
        }
        return result;
    }

    public string Decode(IEnumerable<int> ids)
    {
        var text = string.Concat(ids.Select(id => _itos.TryGetValue(id, out var token) ? token : Unknown));
        return text.Replace("</w>", " ").Trim();
    }
}
