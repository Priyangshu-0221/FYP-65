import React, { useMemo, useState, useEffect } from "react";
import {
  Box,
  Button,
  Container,
  Flex,
  Heading,
  Input,
  Stack,
  Text,
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
  useColorModeValue,
  Icon,
  VStack,
  HStack,
  ScaleFade,
  Fade,
  SlideFade,
  useDisclosure,
} from "@chakra-ui/react";
import { FiUpload, FiSearch, FiAward, FiFileText, FiCheckCircle } from "react-icons/fi";
import "./styles/animations.css";

const API_BASE = "/api";

const buildFormData = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  // Explicitly set process_as_pdf for PDF files
  if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
    formData.append("process_as_pdf", "true");
  }
  return formData;
};

function SkillsList({ skills, isLoading }) {
  const bgGradient = useColorModeValue(
    'linear(to-r, teal.100, blue.100)',
    'linear(to-r, teal.900, blue.900)'
  );

  if (isLoading) {
    return (
      <Wrap spacing={2}>
        {Array(5).fill(0).map((_, i) => (
          <Skeleton key={i} height="28px" width="100px" borderRadius="full" />
        ))}
      </Wrap>
    );
  }

  if (!skills.length) {
    return (
      <Fade in={true}>
        <Text color="gray.500" fontSize="sm">No skills extracted yet. Upload a resume to get started.</Text>
      </Fade>
    );
  }

  return (
    <Wrap spacing={2}>
      {skills.map((skill, index) => (
        <WrapItem key={skill}>
          <ScaleFade in={true} delay={index * 0.1}>
            <Tag 
              size="lg" 
              colorScheme="teal" 
              className="skill-tag"
              boxShadow="sm"
              _hover={{
                transform: 'translateY(-2px)',
                boxShadow: 'md',
              }}
              transition="all 0.2s"
            >
              {skill}
            </Tag>
          </ScaleFade>
        </WrapItem>
      ))}
    </Wrap>
  );
}

function RecommendationGrid({ recommendations, isLoading }) {
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const hoverBorder = useColorModeValue('teal.300', 'teal.500');

  if (isLoading) {
    return (
      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
        {[1, 2, 3, 4].map((_, index) => (
          <Skeleton 
            key={index} 
            height="280px" 
            borderRadius="lg" 
            className="loading-pulse"
            opacity={0.6 + (index * 0.1)}
          />
        ))}
      </SimpleGrid>
    );
  }

  if (!recommendations.length) {
    return (
      <VStack 
        spacing={4} 
        p={8} 
        borderWidth="2px" 
        borderStyle="dashed" 
        borderColor={borderColor}
        borderRadius="lg"
        textAlign="center"
      >
        <Icon as={FiSearch} boxSize={8} color="gray.400" />
        <Text color="gray.500">No recommendations yet. Upload a resume to get personalized internship suggestions.</Text>
      </VStack>
    );
  }

  return (
    <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
      {recommendations.map((internship, index) => (
        <SlideFade in={true} key={internship.id} delay={index * 0.1} offsetY='20px'>
          <Card 
            variant="outline" 
            borderColor={borderColor}
            className="card-hover"
            bg={cardBg}
            height="100%"
            display="flex"
            flexDirection="column"
          >
            <CardHeader pb={2}>
              <Flex justify="space-between" align="flex-start">
                <Box>
                  <Heading size="md" mb={1}>{internship.title}</Heading>
                  <Text fontSize="sm" color="gray.500" mb={2}>
                    {internship.company} • {internship.location}
                  </Text>
                </Box>
                <Badge colorScheme="purple" variant="subtle" px={2} py={1} borderRadius="md">
                  {internship.category}
                </Badge>
              </Flex>
            </CardHeader>
            <CardBody pt={0} pb={4} flexGrow={1}>
              <Text fontSize="sm" color="gray.600" noOfLines={3} mb={4}>
                {internship.description}
              </Text>
              <Box mt="auto">
                <Text fontSize="xs" color="gray.500" mb={2} fontWeight="medium">
                  RELEVANT SKILLS:
                </Text>
                <Wrap spacing={2}>
                  {internship.skills.slice(0, 4).map((skill) => (
                    <WrapItem key={skill}>
                      <Tag size="sm" variant="subtle" colorScheme="teal" borderRadius="full">
                        {skill}
                      </Tag>
                    </WrapItem>
                  ))}
                  {internship.skills.length > 4 && (
                    <WrapItem>
                      <Tag size="sm" variant="subtle" colorScheme="gray" borderRadius="full">
                        +{internship.skills.length - 4} more
                      </Tag>
                    </WrapItem>
                  )}
                </Wrap>
              </Box>
            </CardBody>
            <CardFooter pt={0}>
              <Button
                as={Link}
                href={internship.apply_link}
                isExternal
                colorScheme="teal"
                variant="outline"
                size="sm"
                rightIcon={<Icon as={FiAward} />}
                width="100%"
                className="button-hover"
              >
                View Details & Apply
              </Button>
            </CardFooter>
          </Card>
        </SlideFade>
      ))}
    </SimpleGrid>
  );
}

function App() {
  const toast = useToast();
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [skills, setSkills] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isRecommending, setIsRecommending] = useState(false);
  const [uploadComplete, setUploadComplete] = useState(false);
  
  const bgGradient = useColorModeValue(
    'linear(to-r, teal.500, blue.500)',
    'linear(to-r, teal.600, blue.600)'
  );
  const cardBg = useColorModeValue('white', 'gray.800');
  const borderColor = useColorModeValue('gray.200', 'gray.700');
  const { isOpen: showUploadSection, onOpen: openUploadSection } = useDisclosure({ defaultIsOpen: true });

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setFile(selected);
    setFileName(selected.name);
    setSkills([]);
    setRecommendations([]);
    setUploadComplete(false);
  };

  const uploadResume = async () => {
    if (!file) {
      toast({
        title: "No file selected",
        description: "Please select a PDF or text resume first.",
        status: "warning",
        duration: 4000,
        isClosable: true,
        position: "top"
      });
      return;
    }

    setIsUploading(true);
    try {
      const formData = buildFormData(file);
      console.log('Uploading file:', file.name, 'type:', file.type, 'size:', file.size);
      
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        // Don't set Content-Type header - let the browser set it with the correct boundary
        body: formData,
      });

      const responseData = await response.json();
      console.log('Server response:', responseData);
      
      if (!response.ok) {
        throw new Error(responseData.detail || `Server responded with ${response.status}`);
      }

      if (!responseData.skills) {
        throw new Error("No skills found in the response");
      }

      setSkills(responseData.skills || []);
      setUploadComplete(true);
      openUploadSection();
      
      toast({
        title: "Resume processed successfully",
        description: `We've extracted ${responseData.skills?.length || 0} skills from your resume`,
        status: "success",
        duration: 5000,
        isClosable: true,
        position: "top"
      });
    } catch (error) {
      console.error("Upload error:", error);
      toast({
        title: "Upload failed",
        description: error.message || "We ran into an issue while processing the resume.",
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
