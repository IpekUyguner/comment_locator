import com.github.cliftonlabs.json_simple.JsonArray;
import com.github.cliftonlabs.json_simple.JsonException;
import com.github.cliftonlabs.json_simple.JsonObject;
import com.github.cliftonlabs.json_simple.Jsoner;
import org.apache.commons.lang3.ArrayUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.Arrays.asList;

class LineCode {
  int fileId;
  int lineCount;
  int blockBody;
  String code;
  int label;
  int id;

  public LineCode(Integer fileId, Integer lineCount, Integer blockBody, String code, int label, int id) {
    this.fileId = fileId;
    this.lineCount = lineCount;
    this.blockBody = blockBody;
    this.code = code;
    this.label = label;
    this.id = lineCount;
  }
}

class LineCodeFeatures {
  List<Integer> coded;
  List<Double> embedded;

  public LineCodeFeatures(List<Integer> coded, List<Double> embedded) {
    this.coded = coded;
    this.embedded = embedded;
  }
}

class LineCodeWithFeatures {
  LineCode lineCode;
  LineCodeFeatures lc_feat;

  public LineCodeWithFeatures(LineCode lc, LineCodeFeatures lc_feat) {
    this.lineCode = lc;
    this.lc_feat = lc_feat;
  }
}

class FileSequence {
  List<LineCode> line_codes = new ArrayList<>();

  public int numberLineCode() {
    return (this.line_codes.size());
  }

  public LineCode getLineCode(int index) {
    return this.line_codes.get(index);
  }

  public void addLineCode(LineCode loc) {
    this.line_codes.add(loc);
  }

}



class BlockDataset {
  /**
   * This class is replicated from pytorch implementation from data_processing.py (https://jetbrains.team/p/ml-comment-locator/repositories/comment_locations_suggestions/files/data_processing.py)
   */


  //This number is based on model, cannot be changed.
  public static int MAX_SIZE = 30;  // number of maximum block inside file, also number of maximum sentence inside block, also number of maximum code tokens inside sentence

  List<FileSequence> fileSequences;
  List<String> special_symbols = asList("-unk-");
  List<String> words = new ArrayList<>();
  Dictionary word_2_indexes = new Hashtable();
  Dictionary index_2_word = new Hashtable();
  Dictionary word_2_embeddings = new Hashtable();
  Dictionary idtoFeatures = new Hashtable();
  int[][][][] data;
  float[][][][] dataWeights;
  int[][] sentenceIds;

  public BlockDataset(List<FileSequence> fileSequences, List<String> words, Dictionary vocab_to_id) {
    this.fileSequences = fileSequences;
    this.words = words;
    this.word_2_indexes = vocab_to_id;
  }

  List<Integer> codeText(String text, Dictionary vocab_to_id, boolean ignore_unknown, boolean ignore_special) {
    return this.codeEachWord(text.split(" "), vocab_to_id, ignore_unknown, ignore_special);
  }

  List<Integer> codeEachWord(String[] word_list, Dictionary vocab2Id, boolean ignore_unknown, boolean ignoreSpecial) {
    List<Integer> result = new ArrayList<>();
    for (String word : word_list) {
      if (ignoreSpecial && this.special_symbols.contains(word))
        continue;
      if (vocab2Id.get(word) != null) {
        result.add((Integer) vocab2Id.get(word));
      } else if (!ignore_unknown) {
        result.add((Integer) vocab2Id.get("-unk-"));
      }
    }
    return result;
  }

  List<Double> getAverageEmbedding(List<Integer> wordIds, Dictionary id2Vocab) {
    int foundWord = 0;
    List<Double> sumEmbeddings = Collections.nCopies(300, 0.0);

    for (int wid : wordIds) {
     // System.out.println(wid);
      String token = (String) id2Vocab.get(wid);
      if (token != "-unk-" && this.words.contains(token)) {
        foundWord += 1;
        List<String> list = new ArrayList<String>();
        JsonArray word_2_embeddings = (JsonArray) this.word_2_embeddings.get(token);
        for (int i = 0; i < word_2_embeddings.size(); i++) {
          list.add(word_2_embeddings.getString(i));
        }
        IntStream.range(0, 300).forEach(i -> sumEmbeddings.set(i, sumEmbeddings.get(i) + Double.parseDouble((String) word_2_embeddings.get(i))));
      }

    }
    if (foundWord > 0) {
      int finalFound_word = foundWord;
      IntStream.range(0, 300).forEach(i -> sumEmbeddings.set(i, sumEmbeddings.get(i) / finalFound_word));
    }
    return sumEmbeddings;

  }

  public void createDataset() {
    for (int i = 0; i < this.fileSequences.size(); i++) {
      FileSequence fileSequence = this.fileSequences.get(i);
      for (int j = 0; j < (fileSequence.numberLineCode()); j++) {
        LineCode lineCode = fileSequence.getLineCode(j);
        List<Integer> lineCoded = this.codeText(lineCode.code, this.word_2_indexes, true, true);
        List<Double> lineCodeEmbedded = getAverageEmbedding(lineCoded, this.index_2_word);
        LineCodeFeatures lineCodeFeatures = new LineCodeFeatures(lineCoded, lineCodeEmbedded);
        this.idtoFeatures.put(((lineCode.fileId) + "#" + (lineCode.lineCount)), lineCodeFeatures);
      }
    }

    ArrayList<ArrayList<ArrayList<LineCodeWithFeatures>>> wholeDataSequences = new ArrayList<>();

    int trueNumberBlocks = 0;
    ArrayList<LineCodeWithFeatures> currentSeq = new ArrayList<>();

    for (FileSequence fileSequence : this.fileSequences) {
      if ((wholeDataSequences).size() == 0 || wholeDataSequences.get(wholeDataSequences.size() - 1).size() > 0) //check here is same as python
      {
        ArrayList<ArrayList<LineCodeWithFeatures>> dummy = new ArrayList<>();
        wholeDataSequences.add(dummy);
      }
      for (int i = 0; i < (fileSequence.numberLineCode()); i++) {
        LineCode lineCode = fileSequence.getLineCode(i);
        LineCodeFeatures lineCodeFeatures = (LineCodeFeatures) this.idtoFeatures.get((lineCode.fileId) + "#" + (lineCode.lineCount));
        if (lineCode.blockBody == 1) {
          if (currentSeq.size() > 0) {
            wholeDataSequences.get(wholeDataSequences.size() - 1).add(currentSeq);
            currentSeq = new ArrayList<>();
          }
          if (wholeDataSequences.get(wholeDataSequences.size() - 1).size() >= MAX_SIZE) {
            wholeDataSequences.add(new ArrayList<>());
          }
          trueNumberBlocks += 1;
          currentSeq.add(new LineCodeWithFeatures(lineCode, lineCodeFeatures));
        } else if (lineCode.blockBody == 2 || lineCode.blockBody == 3) {
          if ((currentSeq.size()) < MAX_SIZE)
            currentSeq.add(new LineCodeWithFeatures(lineCode, lineCodeFeatures));
        }
      }

      if ((currentSeq.size()) > 0) {
        wholeDataSequences.get(wholeDataSequences.size() - 1).add(currentSeq);
        currentSeq = new ArrayList<>();
      }
    }

    int totalSequences = (wholeDataSequences).size();
    int countBlocks = 0;
    for (ArrayList<ArrayList<LineCodeWithFeatures>> i : wholeDataSequences) {
      for (ArrayList<LineCodeWithFeatures> j : i) {
        countBlocks += 1;
      }
    }

    System.out.println("num_blocks read = " + (countBlocks));
    assert trueNumberBlocks == countBlocks : "  \"Number of blks read into data %d does not match true number of blocks %d\"";


    //Input to Onnx model is array type
    int[][][][] data = new int[totalSequences][MAX_SIZE][MAX_SIZE][MAX_SIZE];
    float[][][][] dataWeights = new float[totalSequences][MAX_SIZE][MAX_SIZE][MAX_SIZE];
    int[][] targets = new int[totalSequences][MAX_SIZE];
    float[][] targetWeights = new float[totalSequences][MAX_SIZE];
    int[][] sentence_id = new int[totalSequences][MAX_SIZE];

    for (int i = 0; i < (totalSequences); i++)   // for each file
    {
      for (int k = 0; k < (MAX_SIZE); k++)   //  for each block
      {
        for (int j = 0; j < (MAX_SIZE); j++)   //  for each sentence
        {
          int[] sentence_input;
          if (i >= totalSequences || k >= wholeDataSequences.get(i).size() || j >= (wholeDataSequences.get(i).get(k).size())) {
            sentence_input = new int[1];
            sentence_input[0] = (int) this.word_2_indexes.get("-unk-");
          } else {
            List<Integer> arr = wholeDataSequences.get(i).get(k).get(j).lc_feat.coded;
            if (arr.size() >= MAX_SIZE)
              arr = arr.subList(0, MAX_SIZE);
            sentence_input = ArrayUtils.toPrimitive(arr.toArray(new Integer[arr.size()]));
            targets[i][k] = wholeDataSequences.get(i).get(k).get(0).lineCode.label;
            targetWeights[i][k] = 1.0F;
            sentence_id[i][k] = wholeDataSequences.get(i).get(k).get(0).lineCode.lineCount;  // it is for finding the line number in editor to highlight
          }
          int counterDummy = 0;
          for (int element : sentence_input) {
         //   System.out.println("AAAAAAAAAAAAAAAAAAAAAAAAAAAA");
            //    System.out.println(element);
            data[i][k][j][counterDummy] = element;
            dataWeights[i][k][j][counterDummy] = 1.0F;
            counterDummy += 1;
          }
        }
      }

    }
    this.data = data;
    this.dataWeights = dataWeights;
    this.sentenceIds = sentence_id;
  }
}

public class DataPreprocessor {
  private static final int MAX_SIZE = 30;
  Dictionary word_2_indexes = new Hashtable();
  Dictionary index_2_word = new Hashtable();
  Dictionary word_2_embeddings = new Hashtable();
  List<String> words = new ArrayList<>();

  Dictionary getWord2Indexes() {
    return this.word_2_indexes;
  }

  List<String> getWord() {
    return this.words;
  }

  void createWord2Indexes(String input_file) throws FileNotFoundException {
    File myObj = new File(input_file);
    Scanner myReader = new Scanner(myObj);
    while (myReader.hasNextLine()) {
      String[] data = myReader.nextLine().split(" ");
      String word = data[0];
      Integer index = Integer.parseInt(data[1]);
      this.word_2_indexes.put(word, index);
      this.index_2_word.put(index, word);
    }
    myReader.close();

  }

  void readW2VModel(String code_embeddibgs) throws IOException, JsonException {
    Reader reader = Files.newBufferedReader(Paths.get(code_embeddibgs));
    JsonObject parser = (JsonObject) Jsoner.deserialize(reader);
    Set<?> s = parser.keySet();
    Iterator<?> i = s.iterator();
    do {
      String keyWord = i.next().toString();
      JsonArray vector = (JsonArray) parser.get(keyWord);
      this.word_2_embeddings.put(keyWord, vector);
      this.words.add(keyWord);

    } while (i.hasNext());

  }

  List<FileSequence> readCommentLocationFile(String inputFile) {
    System.out.println("Reading comment locations...");
    FileSequence currentFileSequence = new FileSequence();
    int currentFile = -1;
    List<FileSequence> fileSequences = new ArrayList<>();
    String[] lines = inputFile.split("\n");
    int fileId;
    int lineCount;
    int blockBody;
    int label;
    int sentenceId = -1; //to find the sentence in editor
    String code;
    String COMMENT_ONLY_LINE = "COMMENTEMPTY";
    String EMPTY_LINE = "EMPTY";

    for (String data : lines) {   //assumption here is file is not empty
      sentenceId += 1;
      List<String> line = asList(data.trim().split("\t"));
      //fileId, lineCount, blockBody, label, _, _, code, _, _

      fileId = Integer.parseInt(line.get(0));
      lineCount = Integer.parseInt(line.get(1));
      blockBody = Integer.parseInt(line.get(2));
      label = Integer.parseInt(line.get(3));
      code = (line.get(6));

      if (code == COMMENT_ONLY_LINE)
        continue;
      if (code == EMPTY_LINE)
        continue;
      List<String> code_toks = asList(code.split(" ")).subList(0, Math.min((code.split(" ").length), MAX_SIZE));
      code = code_toks.stream().collect(Collectors.joining(" "));
      LineCode lineCode = new LineCode(fileId, lineCount, blockBody, code, label, sentenceId);
      if (fileId != currentFile) {
        if (currentFileSequence.numberLineCode() > 0) {
          fileSequences.add(currentFileSequence);
          currentFileSequence = new FileSequence();
        }
        currentFile = fileId;
      }
      currentFileSequence.addLineCode(lineCode);
    }

    if (currentFileSequence.numberLineCode() > 0) {
      fileSequences.add(currentFileSequence);
    }
    System.out.println("\tThere were " + ((fileSequences.size())) + " file sequences");
    return fileSequences;
  }

}
