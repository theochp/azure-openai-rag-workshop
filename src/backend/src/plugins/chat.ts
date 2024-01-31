import fp from 'fastify-plugin';
import { QdrantClient } from '@qdrant/js-client-rest';
import { ChatOpenAI, OpenAIEmbeddings, type OpenAIChatInput, type OpenAIEmbeddingsParams } from '@langchain/openai';
import { type Message, MessageBuilder, type ChatResponse, type ChatResponseChunk } from '../lib/index.js';
import { type AppConfig } from './config.js';

    
const SYSTEM_MESSAGE_PROMPT = `Assistant helps the Consto Real Estate company customers with support questions regarding terms of service, privacy policy, and questions about support requests. Be brief in your answers.
Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below. If asking a clarifying question to the user would help, ask the question.
For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. Use square brackets to reference the source, for example: [info1.txt]. Don't combine sources, list each source separately, for example: [info1.txt][info2.pdf].
`;

export class ChatService {
  tokenLimit: number = 4000;

  constructor(
    private config: AppConfig,
    private qdrantClient: QdrantClient,
    private chatClient: (options?: Partial<OpenAIChatInput>) => ChatOpenAI,
    private embeddingsClient: (options?: Partial<OpenAIEmbeddingsParams>) => OpenAIEmbeddings,
    private chatGptModel: string,
    private embeddingModel: string,
    private sourcePageField: string,
    private contentField: string,
  ) {}

  async run(messages: Message[]): Promise<ChatResponse> {

    // TODO: implement Retrieval Augmented Generation (RAG) here
    const query = messages[messages.length - 1].content;

    const embeddingsClient = this.embeddingsClient({ modelName: this.embeddingModel });
    const queryVector = await embeddingsClient.embedQuery(query);
    const searchResults = await this.qdrantClient.search(this.config.indexName, {
      vector: queryVector,
      limit: 3,
      params: {
        hnsw_ef: 128,
        exact: false,
      },
    });

    const results: string[] = searchResults.map((result) => {
      const document = result.payload!;
      const sourcePage = document[this.sourcePageField] as string;
      let content = document[this.contentField] as string;
      content = content.replaceAll(/[\n\r]+/g, ' ');
      return `${sourcePage}: ${content}`;
    });
    
    const content = results.join('\n');

    // Get the latest user message (the question), and inject the sources into it
    const userMessage = `${messages[messages.length - 1].content}\n\nSources:\n${content}`;

    const systemMessage = SYSTEM_MESSAGE_PROMPT;

    // Create the messages prompt
    const messageBuilder = new MessageBuilder(systemMessage, this.chatGptModel);
    messageBuilder.appendMessage('user', userMessage);

    // Add the previous messages to the prompt, as long as we don't exceed the token limit
    for (const historyMessage of messages.slice(0, -1).reverse()) {
      if (messageBuilder.tokens > this.tokenLimit) break;
      messageBuilder.appendMessage(historyMessage.role, historyMessage.content);
    }

    const conversation = messageBuilder.messages.map((m) => `${m.role}: ${m.content}`).join('\n\n');
    const thoughts = `Search query:\n${query}\n\nConversation:\n${conversation}`.replaceAll('\n', '<br>');

    const chatClient = this.chatClient({
      temperature: 0.7,
      maxTokens: 1024,
      n: 1,
    });
    const completion = await chatClient.invoke(messageBuilder.getMessages());

    // Return the response in the Chat specification format
    return {
      choices: [
        {
          index: 0,
          message: {
            content: completion.content as string,
            role: 'assistant',
            context: {
              data_points: results,
              thoughts: thoughts,
            },
          },
        },
      ],
    };
  }
}

export default fp(
  async (fastify, options) => {
    const config = fastify.config;

    // Set up Qdrant client
    const qdrantClient = new QdrantClient({
      url: config.qdrantUrl,
      // Port needs to be set explicitly if it's not the default,
      // see https://github.com/qdrant/qdrant-js/issues/59
      port: Number(config.qdrantUrl.split(':')[2])
    });

    const openAIApiKey = '__dummy';

    // Show the OpenAI URL used in the logs
    fastify.log.info(`Using OpenAI at ${config.azureOpenAiUrl}`);

    // Set common options for the clients
    const commonOptions = {
      openAIApiKey,
      azureOpenAIApiVersion: '2023-05-15',
      azureOpenAIApiKey: openAIApiKey,
      azureOpenAIBasePath: `${config.azureOpenAiUrl}/openai/deployments`,
    };


    // Create a getter for the OpenAI chat client
    const chatClient = (options?: Partial<OpenAIChatInput>) =>
      new ChatOpenAI({
        ...options,
        ...commonOptions,
        azureOpenAIApiDeploymentName: config.azureOpenAiChatGptDeployment,
      });

    // Create a getter for the OpenAI embeddings client
    const embeddingsClient = (options?: Partial<OpenAIEmbeddingsParams>) =>
      new OpenAIEmbeddings({
        ...options,
        ...commonOptions,
        azureOpenAIApiDeploymentName: config.azureOpenAiEmbeddingDeployment,
      });
    const chatService = new ChatService(
      config,
      qdrantClient,
      chatClient,
      embeddingsClient,
      config.azureOpenAiChatGptModel,
      config.azureOpenAiEmbeddingModel,
      config.kbFieldsSourcePage,
      config.kbFieldsContent,
    );

    fastify.decorate('chat', chatService);
  },
  {
    name: 'chat',
    dependencies: ['config'],
  },
);

// When using .decorate you have to specify added properties for Typescript
declare module 'fastify' {
  export interface FastifyInstance {
    chat: ChatService;
  }
}

