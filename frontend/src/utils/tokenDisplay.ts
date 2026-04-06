/**
 * Group byte-level BPE tokens into displayable unicode segments.
 *
 * GPT-2 tokenizer splits non-ASCII text (Korean, CJK, emoji) into
 * byte-level tokens that appear as replacement chars (U+FFFD) when
 * decoded individually. This utility merges adjacent broken tokens
 * into their original unicode characters for display.
 */

export interface DisplayToken {
  /** Indices into the original token array */
  indices: number[];
  /** Human-readable display string */
  label: string;
}

const REPLACEMENT = '\ufffd';

export function groupTokensForDisplay(tokens: string[]): DisplayToken[] {
  const result: DisplayToken[] = [];
  let i = 0;

  while (i < tokens.length) {
    const t = tokens[i];

    // If the token is displayable (no replacement chars), use it as-is
    if (!t.includes(REPLACEMENT) && t !== '') {
      result.push({ indices: [i], label: t });
      i++;
      continue;
    }

    // Accumulate broken tokens until we form a valid string
    // Try merging forward: tokens[i..j] decoded together
    let merged = t;
    let j = i + 1;
    while (j < tokens.length && merged.includes(REPLACEMENT)) {
      merged = tokens.slice(i, j + 1).join('');
      if (!merged.includes(REPLACEMENT)) break;
      j++;
    }

    if (!merged.includes(REPLACEMENT) && merged !== '') {
      result.push({
        indices: Array.from({ length: j - i + 1 }, (_, k) => i + k),
        label: merged,
      });
      i = j + 1;
    } else {
      // Still broken — show the raw token
      result.push({ indices: [i], label: t || '?' });
      i++;
    }
  }

  return result;
}
