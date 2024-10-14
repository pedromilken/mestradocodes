# Carregar pacotes necessários
library(readxl)
library(fs)

# Caminhos para o arquivo Excel e a pasta onde as imagens estão armazenadas
excel_file <- "C:/Users/pedro/Downloads/spiral_analysis_results_novo.xlsx"
base_folder <- "C:/Users/pedro/Downloads/UPDRS/RIGIDEZ"
image_folder <- "C:/Users/pedro/Downloads/UPDRS/IMAGENS"  # Pasta onde as imagens estão originalmente

# Ler o arquivo Excel
spiral_data <- read_excel(excel_file)

# Função para reorganizar as imagens com mensagens de debug
reorganize_images <- function(df, base_folder, image_folder) {
  
  # Iterar sobre cada linha do dataframe
  for (i in 1:nrow(df)) {
    image_name <- trimws(as.character(df[i, "FileName"]))  # Obter o nome da imagem da coluna "FileName"
    postura_value <- as.character(df[i, "RIGIDEZ"])  # Garantir que RIGIDEZ é string
    
    # Criar o caminho da pasta de acordo com o valor de RIGIDEZ
    target_folder <- file.path(base_folder, postura_value)
    
    # Se a pasta não existir, criá-la
    if (!dir_exists(target_folder)) {
      dir_create(target_folder)
    }
    
    # Verificar se a imagem já tem a extensão .jpg
    if (!grepl("\\.jpg$", image_name)) {
      image_name <- paste0(image_name, ".jpg")  # Adicionar a extensão se não existir
    }
    
    # Caminho completo da imagem (da pasta de origem)
    image_path <- file.path(image_folder, image_name)  # Usar o nome original da imagem
    
    # Mensagens de debug
    cat("Lendo imagem do Excel:", image_name, "\n")
    cat("Verificando caminho da imagem:", image_path, "\n")
    
    # Mover a imagem para a pasta correspondente ao valor de RIGIDEZ
    if (file_exists(image_path)) {
      file_move(image_path, file.path(target_folder, image_name))  # Preservar o nome original
    } else {
      cat("Imagem não encontrada:", image_name, "\n")
    }
  }
}

# Chamar a função para reorganizar as imagens
reorganize_images(spiral_data, base_folder, image_folder)

