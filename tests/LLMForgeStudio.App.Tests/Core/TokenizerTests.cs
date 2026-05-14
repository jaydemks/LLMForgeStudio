using LLMForgeStudio.App.Core.Tokenization;
using Xunit;

namespace LLMForgeStudio.App.Tests.Core;

public class TokenizerTests
{
    [Fact]
    public void CharacterTokenizer_RoundTrip_IsDeterministic()
    {
        var t = new CharacterTokenizer();
        var cfg = new TokenizerConfig();
        var text = "ciao mondo\nciao";
        t.Train(text, cfg);

        var ids = t.Encode(text);
        var decoded = t.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void WordTokenizer_HandlesPunctuation()
    {
        var t = new WordTokenizer();
        var cfg = new TokenizerConfig { KeepPunctuationAsTokens = true };
        var text = "Ciao, mondo!";
        t.Train(text, cfg);

        var ids = t.Encode(text);
        var decoded = t.Decode(ids);

        Assert.Equal("Ciao, mondo!", decoded);
    }

    [Fact]
    public void ByteLevelBpeTokenizer_RoundTrip_PreservesUtf8()
    {
        var t = new ByteLevelBpeTokenizer();
        var cfg = new TokenizerConfig { MaxMerges = 50, MinFrequency = 1, TargetVocabSize = 512 };
        var text = "Ciao 🌍 — byte test äöü";

        t.Train(text, cfg);
        var ids = t.Encode(text);
        var decoded = t.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void UnigramTokenizer_EncodeDecode_IsStable()
    {
        var t = new UnigramTokenizer();
        var cfg = new TokenizerConfig { MinFrequency = 1, TargetVocabSize = 512 };
        var text = "ciao mondo ciao";

        t.Train(text, cfg);
        var ids = t.Encode(text);
        var decoded = t.Decode(ids);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void WordPieceTokenizer_EncodeDecode_IsStable()
    {
        var t = new WordPieceTokenizer();
        var cfg = new TokenizerConfig { MinFrequency = 1, TargetVocabSize = 512 };
        var text = "training token test";

        t.Train(text, cfg);
        var ids = t.Encode(text);
        var decoded = t.Decode(ids);

        Assert.Equal(text, decoded);
    }
}
