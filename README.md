# <h1 align="center"><font color="gree">RAG with MCP on the Databricks Platform</font></h1>

<font color="pink">Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro</font>


Há soluções que combinam ``RAG`` com ``MCP``, onde o agente IA utiliza o MCP para acessar de maneira estruturada a dados e serviços externos que alimentarão o processo de recuperação e geração de conteúdo, promovendo um sistema autônomo, escalável e eficiente.

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2025/06/How-to-perform-RAG-using-MCP.webp)


## <font color="red">Contextualizando</font>

### <font color="blue">MCP</font>

Model Context Protocol (MCP) é um protocolo que permite expor o contexto e funcionalidades de modelos de linguagem (``LLMs``) como serviços, facilitando a criação de sistemas de geração de texto assistida por recuperação (``RAG``). Um ``MCP server``, como no nosso código, oferece uma interface para um modelo de linguagem com integração de contexto externo (em nosso caso, documentos indexados), melhorando a qualidade e relevância das respostas.


### <font color="blue">server.py</font></font>

O arquivo server.py cria um ``servidor MCP`` que carrega documentos de uma pasta, indexa esses documentos usando um índice vetorial (``VectorStoreIndex``) e expõe uma API para receber consultas e gerar respostas baseadas no modelo OpenAI (``GPT-4.1-nano``). Esse servidor é o backend do sistema ``RAG``.

### <font color="blue">client.py</font></font>

O client.py é um cliente simples que conecta ao servidor ``MCP via HTTP``, enviando as perguntas (queries) para o endpoint ``/predict`` e exibindo as respostas. Ele serve para interagir com o servidor de forma amigável, seja manualmente ou programaticamente.



Em resumo, o servidor (``server.py``) processa e responde às perguntas usando contexto recuperado dos documentos, e o cliente ``(client.py)`` envia essas perguntas e mostra as respostas para o usuário. O ``MCP`` facilita essa arquitetura modular e conectável, especialmente para aplicações ``RAG``, onde o modelo consulta uma base de conhecimento externa para melhorar suas respostas.




Thank God!