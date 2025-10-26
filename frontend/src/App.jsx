import React, { useMemo, useState } from "react";
import {
  Box,
  Button,
  Container,
  Heading,
  Input,
  Stack,
  Text,
  Textarea,
  useToast,
  Tag,
  Wrap,
  WrapItem,
  Divider,
  Skeleton,
  SimpleGrid,
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  Badge,
  Link,
} from "@chakra-ui/react";

const API_BASE = "/api";

const buildFormData = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return formData;
};

function SkillsList({ skills }) {
  if (!skills.length) {
    return <Text color="gray.500">No skills extracted yet.</Text>;
  }
  return (
    <Wrap spacing={2}>
      {skills.map((skill) => (
        <WrapItem key={skill}>
          <Tag size="lg" colorScheme="teal">
            {skill}
          </Tag>
        </WrapItem>
      ))}
    </Wrap>
  );
}

function RecommendationGrid({ recommendations, isLoading }) {
  if (isLoading) {
    return (
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
        {Array.from({ length: 4 }).map((_, index) => (
          <Skeleton key={index} height="200px" borderRadius="md" />
        ))}
      </SimpleGrid>
    );
  }

  if (!recommendations.length) {
    return <Text color="gray.500">No recommendations yet.</Text>;
  }

  return (
    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={4}>
      {recommendations.map((internship) => (
        <Card key={internship.id} variant="outline" borderColor="teal.200">
          <CardHeader>
            <Heading size="md">{internship.title}</Heading>
            <Text fontWeight="medium" color="gray.600">
              {internship.company} • {internship.location}
            </Text>
            <Badge mt={2} colorScheme="purple">
              {internship.category}
            </Badge>
          </CardHeader>
          <CardBody>
            <Text fontSize="sm" color="gray.700">
              {internship.description}
            </Text>
            <Wrap mt={4} spacing={2}>
              {internship.skills.map((skill) => (
                <WrapItem key={skill}>
                  <Tag>{skill}</Tag>
                </WrapItem>
              ))}
            </Wrap>
          </CardBody>
          <CardFooter>
            <Button
              as={Link}
              href={internship.apply_link}
              isExternal
              colorScheme="teal"
              variant="outline"
            >
              View Details
            </Button>
          </CardFooter>
        </Card>
      ))}
    </SimpleGrid>
  );
}

function App() {
  const toast = useToast();
  const [file, setFile] = useState(null);
  const [skills, setSkills] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isRecommending, setIsRecommending] = useState(false);

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setSkills([]);
    setRecommendations([]);
  };

  const uploadResume = async () => {
    if (!file) {
      toast({
        title: "Upload a resume",
        description: "Please select a PDF or text resume first.",
        status: "warning",
        duration: 4000,
        isClosable: true,
      });
      return;
    }

    setIsUploading(true);
    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: buildFormData(file),
      });
      if (!response.ok) {
        throw new Error("Failed to upload resume");
      }
      const data = await response.json();
      setSkills(data.skills ?? []);
      toast({
        title: "Resume processed",
        description: data.skills?.length
          ? "Skills extracted successfully."
          : "No skills detected. You can still request recommendations manually.",
        status: "success",
        duration: 4000,
        isClosable: true,
      });
    } catch (error) {
      console.error(error);
      toast({
        title: "Upload failed",
        description: "We ran into an issue while processing the resume.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsUploading(false);
    }
  };

  const requestRecommendations = async () => {
    if (!skills.length) {
      toast({
        title: "No skills available",
        description: "Upload a resume first or enter skills manually.",
        status: "info",
        duration: 4000,
        isClosable: true,
      });
      return;
    }

    setIsRecommending(true);
    try {
      const response = await fetch(`${API_BASE}/recommend`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ skills, top_k: 6 }),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch recommendations");
      }
      const data = await response.json();
      setRecommendations(data.recommendations ?? []);
    } catch (error) {
      console.error(error);
      toast({
        title: "Recommendation failed",
        description: "Please try again later.",
        status: "error",
        duration: 5000,
        isClosable: true,
      });
    } finally {
      setIsRecommending(false);
    }
  };

  return (
    <Container maxW="6xl" py={10}>
      <Stack spacing={10}>
        <Box textAlign="center">
          <Heading size="xl" color="teal.500">
            Internship Recommender
          </Heading>
          <Text mt={2} color="gray.600">
            Upload your resume to extract skills and receive tailored internship suggestions.
          </Text>
        </Box>

        <Box borderWidth="1px" borderRadius="lg" p={6} boxShadow="sm">
          <Stack spacing={4}>
            <Input type="file" accept=".pdf,.txt" onChange={handleFileChange} />
            <Button
              colorScheme="teal"
              onClick={uploadResume}
              isLoading={isUploading}
              loadingText="Processing"
            >
              Upload & Extract Skills
            </Button>
          </Stack>
        </Box>

        <Box borderWidth="1px" borderRadius="lg" p={6} boxShadow="sm">
          <Heading size="md" mb={4}>
            Extracted Skills
          </Heading>
          <SkillsList skills={skills} />
        </Box>

        <Box borderWidth="1px" borderRadius="lg" p={6} boxShadow="sm">
          <Stack spacing={4}>
            <Heading size="md">Get Recommendations</Heading>
            <Button
              colorScheme="teal"
              onClick={requestRecommendations}
              isLoading={isRecommending}
              loadingText="Fetching"
            >
              Recommend Internships
            </Button>
            <Divider />
            <RecommendationGrid
              recommendations={recommendations}
              isLoading={isRecommending}
            />
          </Stack>
        </Box>
      </Stack>
    </Container>
  );
}

export default App;
