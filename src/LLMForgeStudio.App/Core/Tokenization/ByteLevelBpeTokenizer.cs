using System.Globalization;
using System.Text;

namespace LLMForgeStudio.App.Core.Tokenization;

public sealed class ByteLevelBpeTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _stoi = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _itos = new();
    private readonly List<(string Left, string Right)> _merges = new();
    private const string Unknown = "<unk>";

    public string Name => "Byte-level BPE";
    public TokenizerKind Kind => TokenizerKind.ByteLevelBpe;
    public IReadOnlyList<VocabularyItem> Vocabulary { get; private set; } = Array.Empty<VocabularyItem>();

    public void Train(string text, TokenizerConfig config)
    {
        if (config.LowercaseBeforeTraining) text = text.ToLowerInvariant();
        _merges.Clear();

        var sequence = ToBaseByteTokens(text).ToList();
        if (sequence.Count == 0)
            sequence.Add("00");

        for (var step = 0; step < config.MaxMerges; step++)
        {
            var pairCounts = new Dictionary<(string, string), int>();
            for (var i = 0; i < sequence.Count - 1; i++)
            {
                var pair = (sequence[i], sequence[i + 1]);
                pairCounts[pair] = pairCounts.GetValueOrDefault(pair) + 1;
            }

            var best = pairCounts
                .Where(kv => kv.Value >= config.MinFrequency)
                .OrderByDescending(kv => kv.Value)
                .FirstOrDefault();

            if (best.Key == default) break;

            _merges.Add(best.Key);
            sequence = ApplySingleMerge(sequence, best.Key.Item1, best.Key.Item2).ToList();

            var currentVocab = sequence.Distinct().Count();
            if (currentVocab >= config.TargetVocabSize) break;
        }

        var pieces = sequence.GroupBy(p => p)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key)
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
        var sequence = ToBaseByteTokens(text).ToList();
        foreach (var merge in _merges)
            sequence = ApplySingleMerge(sequence, merge.Left, merge.Right).ToList();

        return sequence.Select(piece => _stoi.TryGetValue(piece, out var id) ? id : 0).ToList();
    }

    public string Decode(IEnumerable<int> ids)
    {
        var bytes = new List<byte>();

        foreach (var id in ids)
        {
            if (!_itos.TryGetValue(id, out var token)) continue;
            if (token == Unknown) continue;

            foreach (var byteToken in token.Split('|', StringSplitOptions.RemoveEmptyEntries))
            {
                if (byte.TryParse(byteToken, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out var value))
                    bytes.Add(value);
            }
        }

        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    private static IEnumerable<string> ToBaseByteTokens(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        foreach (var b in bytes)
            yield return b.ToString("X2", CultureInfo.InvariantCulture);
    }

    private static IEnumerable<string> ApplySingleMerge(List<string> sequence, string left, string right)
    {
        for (var i = 0; i < sequence.Count; i++)
        {
            if (i < sequence.Count - 1 && sequence[i] == left && sequence[i + 1] == right)
            {
                yield return left + "|" + right;
                i++;
            }
            else
            {
                yield return sequence[i];
            }
        }
    }
}
