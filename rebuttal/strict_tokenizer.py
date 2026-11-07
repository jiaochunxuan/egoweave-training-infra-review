"""Keep the frozen Pi05 discretization but fail instead of dropping state tokens."""
import numpy as np
from openpi.models.tokenizer import PaligemmaTokenizer


class StrictPaligemmaTokenizer(PaligemmaTokenizer):
    def tokenize(self, prompt, state=None):
        cleaned=prompt.strip().replace('_',' ').replace('\n',' ')
        if state is not None:
            if not np.isfinite(state).all(): raise ValueError('Nonfinite tokenizer state')
            values=np.digitize(state,bins=np.linspace(-1,1,257)[:-1])-1
            full=f"Task: {cleaned}, State: {' '.join(map(str,values))};\nAction: "
            tokens=self._tokenizer.encode(full,add_bos=True)
        else:
            tokens=self._tokenizer.encode(cleaned,add_bos=True)+self._tokenizer.encode('\n')
        if len(tokens)>self._max_len:
            raise ValueError(f'Complete input requires {len(tokens)} tokens; max_token_len={self._max_len}')
        result=np.zeros(self._max_len,dtype=np.int64);result[:len(tokens)]=tokens
        mask=np.arange(self._max_len)<len(tokens)
        return result,mask
