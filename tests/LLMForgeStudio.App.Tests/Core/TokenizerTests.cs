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
}
