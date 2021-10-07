import com.intellij.openapi.util.NlsSafe;
import com.intellij.psi.PsiFile;
import org.javatuples.Pair;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ReadCFile {
  //This number is based on model, cannot be changed.
  public static int MAX_SIZE = 30;  // number of maximum block inside file, also number of maximum sentence inside block, also number of maximum code tokens inside sentence

  static List<List<Pair<String, Integer>>> chunks(List<Pair<String, Integer>> blockLines) {
    List<List<Pair<String, Integer>>> partitions = new ArrayList<>();

    for (int i = 0; i < blockLines.size(); i += MAX_SIZE) {
      partitions.add(blockLines.subList(i, Math.min(i + MAX_SIZE, blockLines.size())));
    }
    return partitions;
  }

  String createInputTxt(PsiFile psiFile) {
    @NotNull @NlsSafe String whole_file = psiFile.getText();
    String[] lines = whole_file.split("\n");

      //Pair represents code string and its line number
      List<List<org.javatuples.Pair<String, Integer>>> dataBlocks = new ArrayList<>();
      int countLine = -1;
      List<org.javatuples.Pair<String, Integer>> dummy = new ArrayList<>();
      dataBlocks.add(dummy);


      //We create data blocks by dividing them using empty lines.
      // Each code part , which are seperated by empty space line, are assumed coherent itself. (And YES, It is a big assumption !)
      for (String line : lines) {
        countLine += 1;
        if (line.equals(""))
        {
          // empty line
          dummy = new ArrayList<>();
          dataBlocks.add(dummy);
        } else {
          org.javatuples.Pair<String, Integer> pair = new Pair<>(line, countLine);
          dataBlocks.get(dataBlocks.size() - 1).add(pair);
        }
      }

      //If one data blocks is bigger than max size of file(spesified in the model), we divide them smaller subfiles.
      List<List<org.javatuples.Pair<String, Integer>>> finalDataBlocks = new ArrayList<>();
      for (List<org.javatuples.Pair<String, Integer>> blockLines : dataBlocks) {
        if (blockLines.size() >= MAX_SIZE) {
          List<List<org.javatuples.Pair<String, Integer>>> limitedSizeChunks = chunks(blockLines);
          for (List<org.javatuples.Pair<String, Integer>> chunk : limitedSizeChunks) {
            finalDataBlocks.add(chunk);
          }
        } else if (blockLines.size() > 0) {
          finalDataBlocks.add(blockLines);
        }
      }

      //For converting C file as model input file
      int fileId = 0;
      int blockCount = -1;
      String outputFile = "";
      for (List<org.javatuples.Pair<String, Integer>> data_block : finalDataBlocks) {
        int count = 0;
        blockCount += 1;

        for (org.javatuples.Pair<String, Integer> pair : data_block) {
          String line = pair.getValue0();
          int block_body;
          if (count == 0) {
            block_body = 1;
          } else if (count == (data_block).size() - 1) {
            block_body = 3;
          } else {
            block_body = 2;
          }
          if (blockCount == MAX_SIZE) {
            blockCount = -1;
          }
          List<String> list = Arrays.asList(String.valueOf(fileId), String.valueOf(pair.getValue1()), String.valueOf(block_body), "1", "EMPTY", "EMPTY", line, "EMPTY", "EMPTY");
          String result = list.stream().collect(Collectors.joining("\t"));
          result += "\n";
          count += 1;
          outputFile += (result);
        }
      }
      return outputFile;
  }

}
