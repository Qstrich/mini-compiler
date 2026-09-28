# The compiler, one folder per pipeline stage:
#   import -> dialect (tc IR) -> lowering -> runtime
# dialect comes first because the others include its generated headers.
add_subdirectory(dialect)
add_subdirectory(import)
add_subdirectory(lowering)
add_subdirectory(runtime)
