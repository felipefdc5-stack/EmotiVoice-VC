# Cria as seguintes pastas no diretorio do EmotiVoice VC
 .checkpoints- para o EmotiVoice VC possa funcionar com as dempendecias que deve ser usadas para inferencia.
 . Site para baixar o checkpoints: https://huggingface.co/FERNAN89/EmotiVoiceVC/tree/main

 .modules-treining- aqui e clocado os modelo de voz treinados no seed-vc, os arquivo gerados no treino ft_model.pth e config.yml, dentro de uma pasta com o nome do modelo que preferir.

# EmotiVoice VC

Este guia descreve o ambiente Python recomendado e o processo para configurar a virtualenv `.venv`, alem de esclarecer qual arquivo de requisitos utilizar conforme o hardware disponivel.

## Versao do Python

- Utilize **Python 3.10.6** (a versao foi utilizada nas ultimas alteracoes do projeto).
- Verifique a versao instalada executando `python --version` no terminal. O comando deve retornar `Python 3.10.6`.

## Preparando o ambiente virtual

1. Abra um terminal na raiz do projeto (`EmotiVoice VC`).
2. Crie a virtualenv (o diretorio `.venv` sera gerado automaticamente):

   ```powershell
   python -m venv .venv
   ```

3. Ative a virtualenv:
   - PowerShell (Windows): `.\.venv\Scripts\Activate.ps1`
   - Prompt de Comando (Windows): `.\.venv\Scripts\activate.bat`
   - Bash (Linux/macOS): `source .venv/bin/activate`

4. (Opcional, mas recomendado) Atualize o `pip` dentro do ambiente:

   ```powershell
   python -m pip install --upgrade pip
   ```

## Instalando dependencias

- **Somente CPU:** instale os pacotes listados em `requirements.txt`.

  ```powershell
  pip install -r requirements.txt
  ```

- **GPUs NVIDIA:** instale os pacotes otimizados contidos em `requirementsNV.txt`.

  ```powershell
  pip install -r requirementsNV.txt
  ```

  > Observacao: certifique-se de que drivers NVIDIA e bibliotecas CUDA/cuDNN compativeis estejam presentes no sistema antes de usar essa opcao.

## Conferindo a instalacao

Com a virtualenv ativa, execute `pip list` para confirmar se todas as dependencias foram instaladas. Em seguida, prossiga com os scripts do projeto conforme necessario.
