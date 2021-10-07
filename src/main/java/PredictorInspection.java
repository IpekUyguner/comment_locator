import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.github.cliftonlabs.json_simple.JsonException;
import com.intellij.codeInspection.InspectionManager;
import com.intellij.codeInspection.LocalInspectionTool;
import com.intellij.codeInspection.ProblemDescriptor;
import com.intellij.codeInspection.ProblemHighlightType;
import com.intellij.openapi.application.PathManager;
import com.intellij.openapi.editor.Document;
import com.intellij.psi.PsiDocumentManager;
import com.intellij.psi.PsiElement;
import com.intellij.psi.PsiFile;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;


import java.io.IOException;
import java.util.*;


public class PredictorInspection extends LocalInspectionTool {

  public static String PATH_TO_MODEL = "/model.onnx";
  public static String WORD_INDEXES_FILE = "/word_to_indexes.csv";
  public static String WORD_EMBEDDINGS_FILE = "/word_embeddings.txt";

  @Override
  public String getDisplayName() {
    return "Check for comment locations";
  }

  public ProblemDescriptor[] checkFile(@NotNull PsiFile file, @NotNull InspectionManager manager, boolean isOnTheFly) {
    List<ProblemDescriptor> descriptors = new ArrayList<>();

    ReadCFile converter = new ReadCFile();
    String inputFile = converter.createInputTxt(file); // Converts C input file to a data file which will be input to model.
    DataPreprocessor dataProcess = new DataPreprocessor();


    try {
      String ABS_PATH_TO_MODEL = PathManager.getConfigPath() + PATH_TO_MODEL;
      dataProcess.createWord2Indexes(PathManager.getConfigPath() + WORD_INDEXES_FILE);
      dataProcess.readW2VModel(PathManager.getConfigPath() + WORD_EMBEDDINGS_FILE);
      List<FileSequence> file_sequences = null;
      file_sequences = dataProcess.readCommentLocationFile(inputFile);
      List<String> words = dataProcess.getWord();
      Dictionary word2indexes = dataProcess.getWord2Indexes();
      BlockDataset dataset = new BlockDataset(file_sequences, words, word2indexes);
      dataset.createDataset();

      int[][][][] data = dataset.data;
      float[][][][] dataWeights = dataset.dataWeights;
      int[][] sentenceIds = dataset.sentenceIds;
      int[][][][] input = new int[1][][][];
      float[][][][] inputWeights = new float[1][][][];
      int[][] inputSentenceIds = new int[1][];

      OrtEnvironment env = OrtEnvironment.getEnvironment();
      OrtSession session = env.createSession(ABS_PATH_TO_MODEL, new OrtSession.SessionOptions());
      List<Integer> commentLines = new ArrayList();
      List<Double> predictionProbs = new ArrayList();
      System.out.println(data);
      for (int j = 0; j < data.length; j++) {  // If the file is longer than maximum of model specified size for a file, then it is seperated as subfiles to put as input to model. So, for each subfiles:
        input[0] = data[j];
        inputWeights[0] = dataWeights[j];
        inputSentenceIds[0] = sentenceIds[j];

        //Create inputs for Onnx model
        OnnxTensor t1 = OnnxTensor.createTensor(env, input);
        OnnxTensor t2 = OnnxTensor.createTensor(env, inputWeights);
        Map<String, OnnxTensor> map = new HashMap<>();
        map.put("input", t1);
        map.put("input_weights", t2);

        OrtSession.Result results = session.run(map);
        float[][][] labels = (float[][][]) results.get(0).getValue();

        for (int i = 0; i < labels[0].length; i++) {
          // If the probability of being comment worth location >0.5, we display it.
          if (labels[0][i][0] > 0.529 && inputSentenceIds[0][i] != 0) {
            commentLines.add(inputSentenceIds[0][i]);
            predictionProbs.add(Double.valueOf(labels[0][i][0]));
          }
        }
      }

      // Showing predicted results
      @Nullable Document document = PsiDocumentManager.getInstance(file.getProject()).getDocument(file);
      int count = 0;
      for (int line : commentLines) {
        int offset = document.getLineEndOffset(line);

        int start = document.getLineStartOffset(line);
        for (int i = start; i < offset; i++) {
          CharSequence ch = document.getCharsSequence();
          if (!Character.isSpaceChar(ch.charAt(i))) {
            start = i;
            break;
          }
        }

        @NotNull PsiElement define;
        define = file.findElementAt(start);
        OurFix o = new OurFix(define, line);
        @NotNull PsiElement define2;
        define2 = file.findElementAt(offset - 1);

        descriptors.add(manager.createProblemDescriptor(define, define2, "Comment worthy location with confidence level " + predictionProbs.get(count).toString().substring(0, 4), ProblemHighlightType.LIKE_UNKNOWN_SYMBOL, true, new OurFix[]{o}));
        count += 1;
      }

    } catch (IOException e) {
      e.printStackTrace();
    } catch (JsonException e) {
      e.printStackTrace();
    } catch (OrtException e) {
      e.printStackTrace();
    }

    return descriptors.toArray(new ProblemDescriptor[0]);
  }
}