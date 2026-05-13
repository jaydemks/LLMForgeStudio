namespace LLMForgeStudio.App.Core.Training;

public sealed class ModelConfig
{
    public int VocabSize { get; set; } = 65;
    public int BlockSize { get; set; } = 256;
    public int Layers { get; set; } = 6;
    public int Heads { get; set; } = 6;
    public int EmbeddingSize { get; set; } = 384;
    public double Dropout { get; set; } = 0.0;

    public int HeadSize => Heads == 0 ? 0 : EmbeddingSize / Heads;
}
