import com.intellij.codeInspection.LocalQuickFix;
import com.intellij.codeInspection.ProblemDescriptor;
import com.intellij.openapi.editor.Document;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.fileEditor.FileEditorManager;
import com.intellij.openapi.project.Project;
import com.intellij.psi.PsiElement;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

public class OurFix implements LocalQuickFix {
  private PsiElement define;
  private int line;
  public OurFix(PsiElement define, int line ) {
    this.define = define;
    this.line = line;
  }

  @NotNull
  @Override
  public String getName() {
    return "Add your comment here! ";
  }

  @NotNull
  @Override
  public String getFamilyName() {
    return "";
  }

  @Override
  public void applyFix(@NotNull Project project, @NotNull ProblemDescriptor problemDescriptor)
  {
    /**
     * Currently, there is no fix, commented code from below can be used to add comment line to the spesific line.
     */

    FileEditorManager editorManager = FileEditorManager.getInstance(project);
    Editor editor = editorManager.getSelectedTextEditor();
    @Nullable Document document =editor.getDocument();
    /*
    int start = document.getLineStartOffset(this.line);
    String lineText = document.getText().substring(start, document.getLineEndOffset(this.line));
    String trimmed = lineText.trim();
    WriteCommandAction.runWriteCommandAction(project, () ->
     document.insertString(start + (lineText).length() - trimmed.length(), ("//You can write comment here.\n"))
    );
    */
  }
}
