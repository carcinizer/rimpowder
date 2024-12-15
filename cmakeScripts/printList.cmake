function(printPrettyList ListVar PreListCall)
  string(JOIN "\n\t" RIMPOWDER_SOURCES_PRETTY ${${ListVar}})
  message( "${PreListCall}\n\t${RIMPOWDER_SOURCES_PRETTY}")
endfunction()
