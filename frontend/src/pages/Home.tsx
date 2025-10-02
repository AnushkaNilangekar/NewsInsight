import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';
import { useForm } from '@mantine/form';
import { notifications } from '@mantine/notifications';
import {
  Container,
  Title,
  Text,
  Paper,
  Stack,
  TextInput,
  Button,
  Loader,
  Alert,
  Anchor,
  Box,
} from '@mantine/core';
import { IconNews, IconQuestionMark, IconCheck, IconAlertCircle } from '@tabler/icons-react';

type FormData = {
  url1: string;
  url2: string;
  url3: string;
  question: string;
};

export function Home() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState<string[]>([]);

  const form = useForm<FormData>({
    initialValues: {
      url1: '',
      url2: '',
      url3: '',
      question: '',
    },
  });

  const processUrls = useMutation(
    async (urls: string[]) => {
      const validUrls = urls.filter(url => url.trim() !== '');
      if (validUrls.length === 0) {
        throw new Error('Please enter at least one URL');
      }

      const response = await axios.post('http://localhost:5000/api/process', { urls: validUrls });
      return response.data;
    },
    {
      onSuccess: () => {
        notifications.show({
          title: 'Success',
          message: 'Articles processed successfully!',
          color: 'green',
          icon: <IconCheck size={18} />,
        });
      },
      onError: (error: any) => {
        notifications.show({
          title: 'Error',
          message: error.response?.data?.message || 'Failed to process URLs',
          color: 'red',
          icon: <IconAlertCircle size={18} />,
        });
      }
    }
  );

  const askQuestion = useMutation(
    async (question: string) => {
      if (!question.trim()) {
        throw new Error('Please enter a question');
      }

      const response = await axios.post('http://localhost:5000/api/ask', { question });
      return response.data;
    },
    {
      onSuccess: (data) => {
        if (data.status === 'success') {
          setAnswer(data.answer);
          setSources(data.sources || []);
        } else {
          notifications.show({
            title: 'Error',
            message: data.message,
            color: 'red',
            icon: <IconAlertCircle size={18} />,
          });
        }
      },
      onError: (error: any) => {
        notifications.show({
          title: 'Error',
          message: error.response?.data?.message || 'Failed to get answer',
          color: 'red',
          icon: <IconAlertCircle size={18} />,
        });
      }
    }
  );

  const handleProcessUrls = async (values: FormData) => {
    try {
      setIsProcessing(true);
      const urls = [values.url1, values.url2, values.url3].filter(url => url.trim() !== '');
      await processUrls.mutateAsync(urls);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAskQuestion = async (values: FormData) => {
    try {
      setIsProcessing(true);
      await askQuestion.mutateAsync(values.question);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <Box bg="gray.0" style={{ minHeight: '100vh' }} py="xl">
      <Container size="md">
        <Stack gap="xl">
          {/* Header */}
          <Box ta="center" mb="lg">
            <Title order={1} size="h1" mb="xs">
              QuickNews
            </Title>
            <Text size="lg" c="dimmed">
              Get instant answers from news articles
            </Text>
          </Box>

          {/* Usage Info */}
          <Alert variant="light" color="blue" title="How to use this tool" icon={<IconNews />}>
            <Stack gap="xs">
              <Text size="sm">1. Enter news article URLs below</Text>
              <Text size="sm">2. Click "Process URLs" to analyze the articles</Text>
              <Text size="sm">3. Ask a question based on those articles</Text>
              <Text size="sm">4. Get a friendly, sourced answer</Text>
            </Stack>
          </Alert>

          {/* URL Input Section */}
          <Paper shadow="sm" p="lg" radius="md" withBorder>
            <Title order={2} size="h3" mb="md">
              Enter News Article URLs
            </Title>
            <form onSubmit={form.onSubmit(handleProcessUrls)}>
              <Stack gap="md">
                <TextInput
                  label="Article URL 1"
                  placeholder="https://example.com/article-1"
                  {...form.getInputProps('url1')}
                  disabled={isProcessing}
                />
                <TextInput
                  label="Article URL 2 (optional)"
                  placeholder="https://example.com/article-2"
                  {...form.getInputProps('url2')}
                  disabled={isProcessing}
                />
                <TextInput
                  label="Article URL 3 (optional)"
                  placeholder="https://example.com/article-3"
                  {...form.getInputProps('url3')}
                  disabled={isProcessing}
                />
                <Button
                  type="submit"
                  fullWidth
                  leftSection={isProcessing ? <Loader size="xs" color="white" /> : <IconNews size={18} />}
                  disabled={isProcessing}
                  size="md"
                >
                  {isProcessing ? 'Processing...' : 'Process Articles'}
                </Button>
              </Stack>
            </form>
          </Paper>

          {/* Question Section */}
          <Paper shadow="sm" p="lg" radius="md" withBorder>
            <Title order={2} size="h3" mb="md">
              Ask a Question
            </Title>
            <form onSubmit={form.onSubmit(handleAskQuestion)}>
              <Stack gap="md">
                <TextInput
                  placeholder="What would you like to know?"
                  size="md"
                  {...form.getInputProps('question')}
                  disabled={isProcessing}
                  leftSection={<IconQuestionMark size={18} />}
                />
                <Button
                  type="submit"
                  fullWidth
                  color="green"
                  leftSection={isProcessing ? <Loader size="xs" color="white" /> : <IconQuestionMark size={18} />}
                  disabled={isProcessing}
                  size="md"
                >
                  {isProcessing ? 'Thinking...' : 'Ask Question'}
                </Button>
              </Stack>
            </form>

            {/* Answer Display */}
            {answer && (
              <Box mt="xl">
                <Title order={3} size="h4" mb="md">
                  Answer:
                </Title>
                <Paper p="md" bg="gray.0" radius="md">
                  <Text style={{ whiteSpace: 'pre-line' }}>{answer}</Text>
                  
                  {sources.length > 0 && (
                    <Box mt="md" pt="md" style={{ borderTop: '1px solid #dee2e6' }}>
                      <Text size="sm" fw={500} c="dimmed" mb="xs">
                        Sources:
                      </Text>
                      <Stack gap="xs">
                        {sources.map((source, index) => (
                          <Anchor
                            key={index}
                            href={source}
                            target="_blank"
                            rel="noopener noreferrer"
                            size="sm"
                            style={{
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                              display: 'block',
                            }}
                          >
                            {source}
                          </Anchor>
                        ))}
                      </Stack>
                    </Box>
                  )}
                </Paper>
              </Box>
            )}
          </Paper>
        </Stack>
      </Container>
    </Box>
  );
}