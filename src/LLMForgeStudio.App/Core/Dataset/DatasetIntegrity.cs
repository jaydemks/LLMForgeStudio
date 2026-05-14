using System.Security.Cryptography;
using System.Text;

namespace LLMForgeStudio.App.Core.Dataset;

public static class DatasetIntegrity
{
    public static string ComputeSha256FromText(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        var hash = SHA256.HashData(bytes);
        return Convert.ToHexString(hash).ToLowerInvariant();
    }
}
