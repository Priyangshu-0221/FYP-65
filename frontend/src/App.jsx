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
  const [marks, setMarks] = useState({ cgpa: '', percentage: '' });
  
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

      const text = await response.text();
      let responseData;
      try {
        responseData = JSON.parse(text);
      } catch (e) {
        console.error("Non-JSON response:", text);
        throw new Error(`Server returned invalid response: ${text.substring(0, 100)}...`);
      }
      
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
        body: JSON.stringify({ 
          skills, 
          top_k: 6,
          marks: marks,
          skill_count: skills.length
        }),
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
    <Box minH="100vh" w="100vw" position="relative" overflow="hidden" bg="gray.50">
        {/* Animated Gradient Background - Full Screen */}
        <Box
          position="absolute"
          top="0"
          left="0"
          right="0"
          bottom="0"
          zIndex={0}
          bgGradient="linear(to-br, cyan.50, blue.50, purple.50)"
          backgroundSize="400% 400%"
          animation="gradientBG 20s ease infinite"
          sx={{
            "@keyframes gradientBG": {
              "0%": { backgroundPosition: "0% 50%" },
              "50%": { backgroundPosition: "100% 50%" },
              "100%": { backgroundPosition: "0% 50%" },
            },
          }}
        />

        {/* Floating Decorative Orbs - Hidden on mobile for performance */}
        <Box display={{ base: "none", md: "block" }}>
            <Box
            position="absolute"
            top="-10%"
            right="-5%"
            width="500px"
            height="500px"
            bg="cyan.200"
            filter="blur(120px)"
            opacity={0.4}
            zIndex={0}
            animation="float 25s ease-in-out infinite"
            sx={{
                "@keyframes float": {
                "0%": { transform: "translate(0, 0)" },
                "50%": { transform: "translate(-50px, 50px)" },
                "100%": { transform: "translate(0, 0)" },
                },
            }}
            />
            <Box
            position="absolute"
            top="40%"
            left="-10%"
            width="600px"
            height="600px"
            bg="purple.200"
            filter="blur(150px)"
            opacity={0.4}
            zIndex={0}
            animation="floatReverse 30s ease-in-out infinite"
            sx={{
                "@keyframes floatReverse": {
                "0%": { transform: "translate(0, 0)" },
                "50%": { transform: "translate(50px, -50px)" },
                "100%": { transform: "translate(0, 0)" },
                },
            }}
            />
            <Box
            position="absolute"
            bottom="-10%"
            right="20%"
            width="400px"
            height="400px"
            bg="blue.200"
            filter="blur(100px)"
            opacity={0.4}
            zIndex={0}
            animation="float 22s ease-in-out infinite"
            />
        </Box>

      <Container maxW="container.xl" py={{ base: 8, md: 12 }} position="relative" zIndex={1}>
        <VStack spacing={{ base: 8, md: 12 }} align="stretch">
          {/* Header */}
          <VStack spacing={2} textAlign="center" py={{ base: 4, md: 8 }}>
            <Heading 
              as="h1"
              size={{ base: "xl", md: "2xl" }}
              color="gray.800"
              letterSpacing="-0.02em"
              fontWeight="bold"
              lineHeight="shorter"
            >
              Smart CV <Text as="span" color="purple.500">Analyzer</Text>
            </Heading>
            <Text fontSize={{ base: "md", md: "lg" }} color="gray.600" maxW="lg" px={4} fontWeight="medium" letterSpacing="wide">
              AI-POWERED INTERNSHIP MATCHING
            </Text>
          </VStack>

        {/* Upload Section */}
        <ScaleFade in={true} initialScale={0.9}>
          <Box 
            bg={cardBg} 
            p={{ base: 6, md: 10 }} 
            borderRadius={{ base: "xl", md: "2xl" }}
            borderWidth="1px" 
            borderColor={borderColor}
            boxShadow="xl"
            maxW="4xl"
            mx="auto"
          >
            <VStack spacing={{ base: 6, md: 8 }}>
              <VStack spacing={2}>
                 <Icon as={FiFileText} boxSize={{ base: 8, md: 10 }} color="teal.500" />
                 <Heading size={{ base: "md", md: "lg" }} color="gray.700">Upload Your Resume</Heading>
                 <Text fontSize={{ base: "sm", md: "md" }} color="gray.500">Supported formats: PDF, DOCX, TXT</Text>
              </VStack>
              
              <Box 
                w="100%" 
                p={{ base: 6, md: 10 }}
                border="2px dashed" 
                borderColor="teal.200"
                borderRadius="xl"
                bg={useColorModeValue('teal.50', 'gray.700')}
                textAlign="center"
                transition="all 0.2s"
                _hover={{ borderColor: "teal.400", bg: useColorModeValue('teal.100', 'gray.600') }}
              >
                <Input
                  type="file"
                  accept=".pdf,.txt,.doc,.docx"
                  onChange={handleFileChange}
                  display="none"
                  id="resume-upload"
                />
                <label htmlFor="resume-upload">
                  <VStack spacing={3} cursor="pointer">
                    <Icon as={FiUpload} boxSize={{ base: 6, md: 8 }} color="teal.600" />
                    <Button
                      as="span"
                      colorScheme="teal"
                      variant="solid"
                      size={{ base: "md", md: "lg" }}
                      px={{ base: 6, md: 8 }}
                      w={{ base: "full", sm: "auto" }}
                    >
                      Choose File
                    </Button>
                    <Text fontSize="sm" color="gray.500">or drag and drop here</Text>
                  </VStack>
                </label>
                {fileName && (
                  <Fade in={true}>
                    <HStack justify="center" mt={4} spacing={2} bg="white" p={2} borderRadius="md" display="inline-flex" maxW="100%">
                       <Icon as={FiCheckCircle} color="green.500" flexShrink={0} />
                       <Text fontSize="sm" fontWeight="medium" color="gray.700" noOfLines={1}>{fileName}</Text>
                    </HStack>
                  </Fade>
                )}
              </Box>

              <Button
                onClick={uploadResume}
                isLoading={isUploading}
                loadingText="Analyzing..."
                colorScheme="blue"
                size="lg"
                width="full"
                maxW="md"
                disabled={!file}
                borderRadius="xl"
                fontSize="md"
                boxShadow="lg"
                _hover={{ transform: 'translateY(-2px)', boxShadow: 'xl' }}
              >
                Start Analysis
              </Button>
            </VStack>
          </Box>
        </ScaleFade>

        {/* Skills & Academic Section */}
        {uploadComplete && (
          <SlideFade in={true} offsetY="20px">
            <Box 
              bg={cardBg} 
              p={{ base: 5, md: 8 }}
              borderRadius="2xl" 
              borderWidth="1px" 
              borderColor={borderColor}
              boxShadow="xl"
              maxW="4xl"
              mx="auto"
            >
              <VStack spacing={{ base: 6, md: 8 }} align="stretch">
                <Flex direction={{ base: "column", sm: "row" }} justify="space-between" align={{ base: "stretch", sm: "center" }} gap={4}>
                  <VStack align="start" spacing={1}>
                    <Heading size="md" color="teal.600">
                      <Icon as={FiCheckCircle} mr={2} />
                      Analysis Complete
                    </Heading>
                    <Text fontSize="sm" color="gray.500">We extracted {skills.length} skills from your profile</Text>
                  </VStack>
                  <Button
                    onClick={requestRecommendations}
                    isLoading={isRecommending}
                    loadingText="Matching..."
                    colorScheme="teal"
                    size="lg"
                    leftIcon={<FiSearch />}
                    disabled={!skills.length}
                    borderRadius="xl"
                    px={8}
                    boxShadow="md"
                    w={{ base: "full", sm: "auto" }}
                    _hover={{ transform: 'translateY(-2px)', boxShadow: 'lg' }}
                  >
                    Get Recommendations
                  </Button>
                </Flex>
                
                <Box bg="gray.50" p={{ base: 4, md: 5 }} borderRadius="xl">
                   <Text fontSize="xs" fontWeight="bold" color="gray.400" textTransform="uppercase" mb={3}>Extracted Skills</Text>
                   <SkillsList skills={skills} isLoading={false} />
                </Box>

                <Divider borderColor="gray.200" />
                
                <Box>
                   <HStack mb={4} align="center">
                      <Icon as={FiAward} color="purple.500" />
                      <Heading size="md" fontSize="lg">Academic Details</Heading>
                      <Badge colorScheme="purple" variant="subtle">Optional</Badge>
                   </HStack>
                   <SimpleGrid columns={{ base: 1, md: 2 }} spacing={{ base: 4, md: 6 }}>
                      <Box>
                          <Text mb={2} fontWeight="medium" fontSize="sm" color="gray.600">CGPA</Text>
                          <Input 
                              placeholder="e.g. 8.5" 
                              name="cgpa"
                              bg="white"
                              height="48px"
                              borderRadius="lg"
                              focusBorderColor="purple.400"
                              onChange={(e) => setMarks({...marks, cgpa: e.target.value})}
                          />
                      </Box>
                      <Box>
                          <Text mb={2} fontWeight="medium" fontSize="sm" color="gray.600">Percentage</Text>
                          <Input 
                              placeholder="e.g. 85" 
                              name="percentage"
                              bg="white"
                              height="48px"
                              borderRadius="lg"
                              focusBorderColor="purple.400"
                              onChange={(e) => setMarks({...marks, percentage: e.target.value})}
                          />
                      </Box>
                   </SimpleGrid>
                </Box>
              </VStack>
            </Box>
          </SlideFade>
        )}


        {/* Recommendations Section */}
        {(recommendations.length > 0 || isRecommending) && (
          <SlideFade in={true} offsetY="30px">
            <Box 
              bg="transparent" 
              pt={4}
            >
              <VStack spacing={6} align="stretch">
                <HStack justify="center" mb={4}>
                   <Heading size={{ base: "xl", md: "2xl" }} color="gray.800" textAlign="center">
                    Top Career Matches
                  </Heading>
                </HStack>
                
                <RecommendationGrid 
                  recommendations={recommendations} 
                  isLoading={isRecommending} 
                />
              </VStack>
            </Box>
          </SlideFade>
        )}
      </VStack>
    </Container>
    </Box>
  );
}

export default App;
