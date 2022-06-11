import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.*;

public class HMM {
    private double lambda;

    //Constructor
    public HMM(double lambda){
        this.lambda = lambda;
    }

    //Training
    public void train(String trainFile, String modelFile){
        try{
            List<String> tagSet = new ArrayList<>();
            HashMap<String, Double> probTags = new HashMap<String,Double>();
            HashMap<String, Double> probEmission = new HashMap<String, Double>();
            HashMap<String, Double> probTransition = new HashMap<String, Double>();

            //Intermediate tables
            HashMap<String, Integer> countTags = new HashMap<String, Integer>();
            HashMap<String, Integer> countWords = new HashMap<String, Integer>();
            HashMap<String,Integer> countEmission = new HashMap<String,Integer>();
            HashMap<String,Integer> countTransition = new HashMap<String,Integer>();
            HashSet<String> unknownWords = new HashSet<String>();
            HashMap<String,Integer> countEmissionUNKA = new HashMap<String,Integer>();

            //File Reading
            BufferedReader br = new BufferedReader(new FileReader(trainFile));

            String line = "";
            int sizeCorpus = 0, c = 0;

            //Count words, tags, emission and transitions
            while((line = br.readLine()) != null){
                c++;
                String[] strInLine = line.trim().split("[/| ]+");

                for (int i = 0; i < strInLine.length; i += 2) {
                    // Count words
                    if (countWords.containsKey(strInLine[i])) {
                        countWords.put(strInLine[i], countWords.get(strInLine[i]) + 1);
                    } else
                        countWords.put(strInLine[i], 1);
                    // Count tags
                    if (countTags.containsKey(strInLine[i + 1])) {
                        countTags.put(strInLine[i + 1], countTags.get(strInLine[i + 1]) + 1);
                    } else{
                        countTags.put(strInLine[i + 1], 1);
                    }

                    // Count emission
                    String wordTag = strInLine[i] + " " + strInLine[i + 1];

                    if (countEmission.containsKey(wordTag))
                        countEmission.put(wordTag, countEmission.get(wordTag) + 1);
                    else
                        countEmission.put(wordTag, 1);
                    // Count tag transitions: t_i|t_{i-1}
                    if (i > 0) { // Skips i==0, since transition needs two tags
                        String tagTag = strInLine[i + 1] + " " + strInLine[i - 1];

                        if (countTransition.containsKey(tagTag))
                            countTransition.put(tagTag, countTransition.get(tagTag) + 1);
                        else
                            countTransition.put(tagTag, 1);
                    }
                    // Deal with "START" and "END"
                    if (countTags.containsKey("START")) {
                        countTags.put("START", countTags.get("START") + 1);
                    } else
                        countTags.put("START", 1);
                    if (countTags.containsKey("END"))
                        countTags.put("END", countTags.get("END") + 1);
                    else
                        countTags.put("END", 1);

                    String specialTagTag = strInLine[1] + " " + "START";

                    if (countTransition.containsKey(specialTagTag))
                        countTransition.put(specialTagTag, countTransition.get(specialTagTag) + 1);
                    else
                        countTransition.put(specialTagTag, 1);

                    specialTagTag = "END" + " " + strInLine[strInLine.length - 1];
                    if (countTransition.containsKey(specialTagTag))
                        countTransition.put(specialTagTag, countTransition.get(specialTagTag) + 1);
                    else
                        countTransition.put(specialTagTag, 1);

                    // Accumulate to calculate the size of corpus
                    sizeCorpus += strInLine.length / 2;
                }
                // Identify the words that occur less than 1 times, and hash them into the unknown word set
                Iterator<Map.Entry<String,Integer>> itWord = countWords.entrySet().iterator();
                while(itWord.hasNext()) {
                    Map.Entry<String,Integer> pair = itWord.next();
                    if(pair.getValue()<1)
                        unknownWords.add(pair.getKey());
                }
            }
            // Calculate the probabilities of words given tags (emission probabilities)
            Iterator<Map.Entry<String,Integer>> itWordTag = countEmission.entrySet().iterator();

            while(itWordTag.hasNext()) {
                Map.Entry<String,Integer> pair = itWordTag.next();
                String [] wordTag = pair.getKey().split("[ ]+");
                if(unknownWords.contains(wordTag[0])) { // Unknown words: UNKA
                    String wordTagUNKA = "UNKA" + " " + wordTag[1];
                    if(countEmissionUNKA.containsKey(wordTagUNKA))
                        countEmissionUNKA.put(wordTagUNKA,countEmissionUNKA.get(wordTagUNKA)+pair.getValue());
                    else
                        countEmissionUNKA.put(wordTagUNKA,pair.getValue());
                } else { // Known words
                    probEmission.put(pair.getKey(),1.0*pair.getValue()/countTags.get(wordTag[1])); //word and tag prob in training data
                }

            }
            Iterator<Map.Entry<String,Integer>> itWordTagUNKA = countEmissionUNKA.entrySet().iterator();
            while(itWordTagUNKA.hasNext()) {
                Map.Entry<String,Integer> pair = itWordTagUNKA.next();
                String [] wordTagUNKA = pair.getKey().split("[ ]+");
                probEmission.put(pair.getKey(),1.0*pair.getValue()/countTags.get(wordTagUNKA[1]));
            }

            // Calculate the tag probabilities and tag transition probabilities
            Iterator<Map.Entry<String,Integer>> itTag = countTags.entrySet().iterator();
            int numTags = countTags.size();
            String [] str = new String[numTags];
            int k = 0;
            while(itTag.hasNext()) {
                Map.Entry<String,Integer> pair = itTag.next();
                str[k] = pair.getKey();
                tagSet.add(str[k]);
                probTags.put(str[k],1.0*pair.getValue()/sizeCorpus);
                k++;

            }

            Iterator<Map.Entry<String,Integer>> itTagTag = countTransition.entrySet().iterator();
            while(itTagTag.hasNext()) {
                Map.Entry<String,Integer> pair = itTagTag.next();
                String [] tagTag = pair.getKey().split("[ ]+");
                double probTagTagML = 1.0*pair.getValue()/countTags.get(tagTag[1]);
                double prob = lambda*(probTagTagML) + (1-lambda)*probTags.get(tagTag[0]);
                probTransition.put(pair.getKey(),prob);
            }
            //Creating a JSONObject object
            JSONObject jsonObject = new JSONObject();

            jsonObject.put("TagSet", tagSet);
            jsonObject.put("ProbTags", probTags);
            jsonObject.put("ProbEmission", probEmission);
            jsonObject.put("ProbTransition", probTransition);

            FileWriter file = new FileWriter(modelFile);
            file.write(jsonObject.toJSONString());
            file.close();

            br.close();

        }
        catch (Exception e){
            System.out.println("Something went wrong!");
        }

    }
    // Finding the most probable path: Viterbi algorithm
    public double viterbi(String [] line, String [] tags, String modelFile) throws Exception{

        //Creating a JSONParser object
        JSONParser jsonParser = new JSONParser();

        //Parsing the contents of the JSON file
        JSONObject model = (JSONObject) jsonParser.parse(new FileReader(modelFile));

        List<String> tagSet = (List<String>) model.get("TagSet");
        HashMap<String,Double> probEmission = (HashMap<String,Double>) model.get("ProbEmission");
        HashMap<String,Double> probTransition = (HashMap<String,Double>) model.get("ProbTransition");

        // Initialization
        int N = line.length;
        int M = tagSet.size();

        double vScore[][] = new double[M][N];
        int backPTR[][] = new int[M][N];
        for(int i=0;i<M;i++) {
            String wordTag = line[0] + " " + tagSet.get(i);
            String tagTag = tagSet.get(i) + " " + "START";

            double probWordTag = probEmission.containsKey(wordTag)?probEmission.get(wordTag):0.0;
            double probTagTag = probTransition.containsKey(tagTag)?probTransition.get(tagTag):0.0;

            vScore[i][0] = Math.log(probWordTag) + Math.log(probTagTag);
            backPTR[i][0] = -1;
        }

        // Iteration
        for(int j =1;j<N;j++) {
            for(int i=0;i<M;i++) {
                vScore[i][j] = Double.NEGATIVE_INFINITY;

                for(int k=0;k<M;k++) {
                    String tagTag = tagSet.get(i) + " " + tagSet.get(k);
                    String wordTag = line[j] + " " + tagSet.get(i);

                    double probTagTag = probTransition.containsKey(tagTag)?probTransition.get(tagTag):0.0;
                    double probWordTag = probEmission.containsKey(wordTag)?probEmission.get(wordTag):0.0;
                    double score = vScore[k][j-1] + Math.log(probTagTag) + Math.log(probWordTag);

                    if(score>vScore[i][j]) {
                        vScore[i][j] = score;
                        backPTR[i][j] = k;
                    }
                }
            }
        }

        // Get the path and log probability
        double logProb = Double.NEGATIVE_INFINITY;
        int path[] = new int[N];
        for(int i=0;i<M;i++) {
            String tagTag = "END" + " " + tagSet.get(i);
            double probTagTag = probTransition.containsKey(tagTag)?probTransition.get(tagTag):0.0;
            double score = vScore[i][N-1] + Math.log(probTagTag);

            if(score>logProb) {
                logProb = score;
                path[N-1] = i;
            }
        }

        tags[N-1] = tagSet.get(path[N-1]);
        for(int j=N-2;j>=0;j--) {
            path[j] = backPTR[path[j+1]][j+1];
            tags[j] = tagSet.get(path[j]);
        }

        return logProb;
    }



    public String test(String input, String modelFile) throws Exception {
        String output = "";

        try{
            String [] strInLine = input.trim().split("[ ]+"); //RE for space
            String [] tags = new String[strInLine.length];
            double logProb = viterbi(strInLine,tags, modelFile);
            for(int i=0;i<tags.length;i++) {
                output += strInLine[i] + "/" + tags[i] + " ";
            }
        }
        catch (Exception e) {
            System.out.println("Something went wrong!");
        }
        return output;
    }
}
