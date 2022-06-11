public class test {
    public static void main(String[] args) throws Exception{

        String trainFile = "src/main/resources/train.txt";
        String modelFile = "src/main/resources/train.json";
        double tagvalue = 0.5;

        HMM tagger = new HMM(tagvalue);
        tagger.train(trainFile, modelFile);

        String input = "ဒီနေ့ ဘာ ရုပ်ရှင်ကား ပြ နေ လဲ";
        String result = tagger.test(input, modelFile);

        System.out.println("Input: " + input);
        System.out.println("Output: " + result);


    }
}
