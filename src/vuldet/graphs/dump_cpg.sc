// Nguyễn Tâm Bình - 2324801030003
import java.io.PrintWriter

def esc(s: String): String = {
  val sb = new StringBuilder
  s.foreach {
    case '"' => sb.append("\\\"")
    case '\\' => sb.append("\\\\")
    case '\n' => sb.append("\\n")
    case '\r' => sb.append("\\r")
    case '\t' => sb.append("\\t")
    case c if c < ' ' => sb.append(f"\\u${c.toInt}%04x")
    case c => sb.append(c)
  }
  sb.toString
}

@main def exec(cpgFile: String, outFile: String, maxCode: Int = 200) = {
  importCpg(cpgFile)
  val out = new PrintWriter(outFile, "UTF-8")
  cpg.method.isExternal(false).nameNot("<global>").foreach { m =>
    val nodes = m.ast.l
    val ids = nodes.map(_.id).toSet
    val nodeJson = nodes.map { n =>
      val code = esc(n.code.take(maxCode))
      s"""[${n.id},"${n.label}","$code",${n.lineNumber.getOrElse(-1)}]"""
    }
    val edges = scala.collection.mutable.ArrayBuffer[String]()
    nodes.foreach { n =>
      n._astOut.filter(x => ids.contains(x.id)).foreach(d => edges += s"""[${n.id},${d.id},"AST"]""")
      n._cfgOut.filter(x => ids.contains(x.id)).foreach(d => edges += s"""[${n.id},${d.id},"CFG"]""")
      n._cdgOut.filter(x => ids.contains(x.id)).foreach(d => edges += s"""[${n.id},${d.id},"CDG"]""")
      n._reachingDefOut.filter(x => ids.contains(x.id)).foreach(d => edges += s"""[${n.id},${d.id},"DDG"]""")
    }
    out.println(s"""{"file":"${esc(m.filename)}","name":"${esc(m.name)}","line":${m.lineNumber.getOrElse(-1)},"nodes":[${nodeJson.mkString(",")}],"edges":[${edges.mkString(",")}]}""")
  }
  out.close()
}
