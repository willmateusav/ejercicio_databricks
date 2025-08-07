from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, BooleanType
from typing import List, Optional, Dict

def preprocesar_colaboradores(
    df: DataFrame,
    diccionario_equivalencias: Optional[Dict[str, str]] = None,
    columna_procesar: str = "ProcesarMotor",
    valor_procesar: int = 1,
    columnas_lower: Optional[List[str]] = None,
    columnas_int: Optional[List[str]] = None,
    columnas_array_int: Optional[List[str]] = None,
    columnas_array_string: Optional[List[str]] = None,
    booleanos_a_string: Optional[Dict[str, str]] = None,
    int_a_string: Optional[List[str]] = None,
    riesgos: Optional[List[str]] = None
) -> DataFrame:
    """
    Preprocesa un DataFrame de colaboradores aplicando renombramiento, transformaciones de tipo,
    limpieza de strings y generación de nuevas columnas.

    Parámetros:
    -----------
    df : pyspark.sql.DataFrame
        DataFrame original de colaboradores.
    
    diccionario_equivalencias : dict, opcional
        Diccionario con las equivalencias de nombres de columnas a aplicar.
    
    columna_procesar : str, opcional
        Columna usada como condición para procesar el registro. Default = "ProcesarMotor".
    
    valor_procesar : int, opcional
        Valor esperado en la columna de procesamiento. Default = 1.
    
    columnas_lower : list[str], opcional
        Lista de columnas a transformar en minúsculas.
    
    columnas_int : list[str], opcional
        Lista de columnas a convertir a tipo entero.

    columnas_array_int : list[str], opcional
        Lista de columnas con strings separados por coma que se convertirán en arrays de enteros.

    booleanos_a_string : dict, opcional
        Diccionario de columnas booleanas a convertir a string con capitalización.

    riesgos : list[str], opcional
        Lista de columnas que representan riesgos y se agruparán en un array.

    Retorna:
    --------
    pyspark.sql.DataFrame
        DataFrame transformado y listo para usar.
    """

    if diccionario_equivalencias is None:
        diccionario_equivalencias = {
            'Cargo': 'Cargo_',
            'NivelCargo': 'Cargo',
            'Antiguedad': 'Antiguedad_',
            'AntiguedadAnios': 'Antiguedad',
            'CiudadLabora': 'Ciudad',
            'PersonaMayor': 'AdultoMayor',
            'Gestante': 'Gestantes',
            'TieneMascota': 'Mascota',
            'Discapacidad': 'PersonasEspeciales',
            'CountSMLV': 'Ingresos',
            'FechaIngresoEmpresa': 'FechaIngreso',
            'TiposMascotas': 'TipoMascota',
            'IdEmpresaPlataforma': 'IdProveedor'
        }

    if columnas_lower is None:
        columnas_lower = [
            "Ciudad", "TipoFamilia", "TipoMascota", "categoriaAfiliacion", "Cargo", "Genero",
            "TipoContrato", "TipoRemuneracion", "RiesgoCardiovascular", "RiesgoOsteomuscular", "RiesgoPsicosocial"
        ]

    if columnas_int is None:
        columnas_int = ["Edad", "Ingresos", "Antiguedad"]

    if columnas_array_int is None:
        columnas_array_int = ["EdadHijos", "EdadBeneficiarios"]

    if columnas_array_string is None:
        columnas_array_string = ["TipoMascota", "NurGrupos"]

    if booleanos_a_string is None:
        booleanos_a_string = {
            "Mascota": "Mascota",
            "AdultoMayor": "AdultoMayor"
        }
    
    if int_a_string is None:
        int_a_string = ["IdEmpresaPlataforma"]

    if riesgos is None:
        riesgos = ["RiesgoCardiovascular", "RiesgoOsteomuscular", "RiesgoPsicosocial"]

    # Filtro por columna_procesar
    df = df.filter(F.col(columna_procesar) == valor_procesar)

    # Renombrar columnas
    for col_original, nueva_col in diccionario_equivalencias.items():
        if col_original in df.columns:
            df = df.withColumnRenamed(col_original, nueva_col)

    # Transformar columnas a minúsculas
    for col in columnas_lower:
        if col in df.columns:
            df = df.withColumn(col, F.lower(F.col(col)))

    # Convertir columnas a int
    for col in columnas_int:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("int"))

    # Convertir columnas separadas por coma en array<int>
    for col in columnas_array_int:
        if col in df.columns:
            df = df.withColumn(col, F.expr(f"transform(split({col}, ','), x -> cast(trim(x) as int))"))

    # Convertir booleanos a string capitalizado
    for col, alias in booleanos_a_string.items():
        if col in df.columns:
            df = df.withColumn(alias, F.initcap(F.col(col).cast(StringType())))

    # Convertir TipoMascota en array<string>
    for col in columnas_array_string:
        if col in df.columns:
            df = df.withColumn(col, F.expr(f"transform(split({col}, ','), x -> trim(x))"))

    # Crear columna ProgramaVigilanciaEpidemiologica como array
    if all(r in df.columns for r in riesgos):
        df = df.withColumn(
            "ProgramaVigilanciaEpidemiologica",
            F.array(*[F.col(r) for r in riesgos])
        )

    # Convertimos las columnas especificadas en `int_a_string` a tipo string
    for col in int_a_string:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("string"))

    return df

# --------------------------------------------
# Ejemplo de uso:
# --------------------------------------------
# Suponiendo que ya tienes cargados el DataFrame df_colaboradores, pues aplicar la función así:
# Aplicación # 1:
# df_colaboradores_preprocesado = preprocesar_colaboradores(df_colaboradores)

# Aplicación # 2:
# df_colaboradores_preprocesado = preprocesar_colaboradores(
#     df_colaboradores,
#     columnas_lower=["Ciudad", "Cargo"],
#     columnas_int=["Edad"],
#     columnas_array_int=["EdadHijos"],
#     booleanos_a_string={"Mascota": "TieneMascota"},
#     riesgos=["RiesgoPsicosocial"]
# )

def construir_metadata_segmentador(
    spark_session=None,
    atributos_rango: list = None,
    atributos_lista: list = None,
    atributos_igual: list = None
):
    """
    Construye un DataFrame con las reglas de segmentación clasificadas por tipo.

    Args:
        spark_session (SparkSession, opcional): Sesión Spark. En Databricks no es necesario pasarla.
        atributos_rango (list, opcional): Lista de nombres de columnas con tipo 'rango'.
        atributos_lista (list, opcional): Lista de nombres de columnas con tipo 'lista'.
        atributos_igual (list, opcional): Lista de nombres de columnas con tipo 'igual'.

    Returns:
        DataFrame: Un DataFrame de Spark con columnas ['atributo', 'tipo_regla'].
    """
    if spark_session is None:
        spark_session = spark  # Usar la sesión global en Databricks

    if atributos_rango is None:
        atributos_rango = ['Edad', 'EdadHijos', 'EdadBeneficiarios', 'Ingresos', 'Antiguedad']
    if atributos_lista is None:
        atributos_lista = ['Genero', 'TipoContrato', 'Cargo', 'Ciudad', 'TipoFamilia', 'TipoMascota',
                           'categoriaAfiliacion', 'TipoRemuneracion', 'ProgramaVigilanciaEpidemiologica', 'NurGrupos']
    if atributos_igual is None:
        atributos_igual = ['AdultoMayor', 'Gestantes', 'Mascota', 'afiliacionCajaCompensacion',
                           'PersonasEspeciales', 'beneficiarioPbsCompensar', 'FechaIngreso',
                           'afiliacionPbsCompensar', 'beneficiarioPcCompensar', "IdProveedor"]

    reglas = (
        [Row(atributo=atributo, tipo_regla="rango") for atributo in atributos_rango] +
        [Row(atributo=atributo, tipo_regla="lista") for atributo in atributos_lista] +
        [Row(atributo=atributo, tipo_regla="igual") for atributo in atributos_igual]
    )

    df = spark_session.createDataFrame(reglas)
    return df

# ------------------------------------------
# EJEMPLOS DE USO
# ------------------------------------------

# Ejecución con parámetros por defecto
# metadata_segmentador = construir_metadata_segmentador()
# display(metadata_segmentador)

# Ejecución con listas personalizadas
# metadata_segmentador_custom = construir_metadata_segmentador(
#     atributos_rango=["Edad", "Ingresos"],
#     atributos_lista=["Ciudad"],
#     atributos_igual=["AdultoMayor", "Gestantes"]
# )
# display(metadata_segmentador_custom)

def preprocesar_experiencias(df: DataFrame, reglas: List[tuple]) -> DataFrame:
    """
    Preprocesa el DataFrame de experiencias aplicando transformaciones según el tipo de regla:
    - "rango": convierte valores como "18a25" en una lista de enteros [18, 19, ..., 25].
    - "igual": reemplaza el valor "_Sindato" por None.
    - "lista": convierte cadenas separadas por comas en listas, limpiando espacios y pasando a minúscula.

    Parámetros:
    ----------
    df : DataFrame
        DataFrame de experiencias a procesar.
    reglas : List[tuple]
        Lista de tuplas (atributo, tipo_regla) que define cómo transformar cada atributo.

    Retorna:
    -------
    DataFrame
        El DataFrame con las columnas transformadas según las reglas.
    """

    # Filtro inicial para conservar solo las filas activas
    df = (
        df
        .filter(F.col("Estado") == "ACTIVO")
        .select([
            # Reemplaza "_Sindato" por None para columnas string, deja las demás igual
            F.when(F.col(c) == "_Sindato", F.lit(None)).otherwise(F.col(c)).alias(c)
            if dtype == "string" else F.col(c)
            for c, dtype in df.dtypes
        ])
        .withColumn(
            "IdProveedor",
            F.when(F.col("IdProveedor").cast("string") == "0", None)
            .otherwise(F.col("IdProveedor").cast("string"))
        )
    )

    # Aplica las transformaciones según el tipo de regla
    for atributo, tipo in reglas:

        if tipo == "rango":
            # Si el tipo es rango (como "18a25"), se convierte a una lista de enteros
            col_lista = f"{atributo}_lista"
            df = df.withColumn(
                col_lista,
                F.expr(f"""
                    filter(
                        aggregate(
                            transform(
                                split({atributo}, ','),
                                x -> sequence(
                                    int(trim(split(x, 'a')[0])),
                                    int(trim(split(x, 'a')[1]))
                                )
                            ),
                            cast(array() as array<int>),
                            (acc, x) -> concat(acc, x)
                        ),
                        x -> x is not null
                    )
                """)
            )

        elif tipo == "igual":
            # Si el tipo es igual, solo se reemplaza "_Sindato" por None
            df = df.withColumn(
                atributo,
                F.when(F.col(atributo) == "_Sindato", None).otherwise(F.col(atributo))
            )

        elif tipo == "lista":
            # Si el tipo es lista, se transforma en una lista limpia y en minúsculas
            df = df.withColumn(
                atributo,
                F.expr(f"transform(split(lower({atributo}), ','), x -> trim(x))")
            )

    return df

# --------------------------------------------
# Ejemplo de uso:
# --------------------------------------------
# Realizas el llamado de la tabla experiencias a través de la siguiente línea de código:
# df_experiencias = df_experiencias = spark.table("slv_enrich_dev.sch_plataforma_bienestar.plataforma_experiencias")
# 
# reglas = [
#     ("Edad", "rango"),
#     ("Genero", "lista"),
#     ("Gestantes", "igual")
# ]
# 
# df_preprocesado = preprocesar_experiencias(df_experiencias, reglas)
# df_preprocesado.display()

def generar_condiciones_join(
    usuarios_df: DataFrame, 
    experiencias_df: DataFrame, 
    reglas_df: DataFrame
) -> List:
    """
    Genera una lista de condiciones para realizar un join entre dos DataFrames
    (`usuarios_df` y `experiencias_df`), con base en un conjunto de reglas.

    Las reglas definen el tipo de comparación a realizar por atributo:
    - 'igual': compara directamente los valores.
    - 'rango': compara si hay intersección entre arrays de enteros (ej. rangos de edad).
    - 'lista': compara si hay intersección entre arrays de strings o si contiene "cualquiera".

    Args:
        usuarios_df (DataFrame): DataFrame con los datos de los usuarios.
        experiencias_df (DataFrame): DataFrame con las experiencias a comparar.
        reglas_df (DataFrame): DataFrame con columnas 'atributo' y 'tipo_regla'.

    Returns:
        List: Lista de condiciones (expresiones booleanas) que pueden ser usadas en un filtro o join.
    """
    condiciones = []

    for row in reglas_df.collect():
        attr, tipo = row["atributo"], row["tipo_regla"]

        if tipo == "igual":
            # Compara directamente los valores entre usuarios y experiencias, 
            # permitiendo que el valor en experiencias sea nulo
            condiciones.append(
                (usuarios_df[attr] == experiencias_df[attr]) | experiencias_df[attr].isNull()
            )

        elif tipo == "rango":
            # Compara si hay intersección entre arrays de enteros (por ejemplo, rangos de edad)
            # Si el valor en usuarios no es un array, se convierte en uno para permitir la comparación
            es_array = isinstance(usuarios_df.schema[attr].dataType, ArrayType)
            nuevo_usuario = usuarios_df[attr] if es_array else F.array(usuarios_df[attr])
            condiciones.append(
                F.arrays_overlap(nuevo_usuario, experiencias_df[f"{attr}_lista"]) |
                experiencias_df[f"{attr}_lista"].isNull()
            )

        elif tipo == "lista":
            # Compara si hay intersección entre listas de strings
            # Si la experiencia tiene "cualquiera", se considera verdadero si hay un valor en usuarios
            es_array = isinstance(usuarios_df.schema[attr].dataType, ArrayType)
            nuevo_usuario = usuarios_df[attr] if es_array else F.array(usuarios_df[attr])
            condiciones.append(
                F.when(
                    (F.array_contains(experiencias_df[attr], "cualquiera")) & (usuarios_df[attr].isNotNull()),
                    True
                ).otherwise(
                    F.arrays_overlap(nuevo_usuario, experiencias_df[attr]) |
                    experiencias_df[attr].isNull()
                )
            )

        else:
            # Lanza error si el tipo de regla no es reconocido
            raise ValueError(f"Tipo de regla no soportado: {tipo}")

    return condiciones


# --------------------------------------------
# Ejemplo de uso de la función
# --------------------------------------------
# Suponiendo que ya tienes cargados tres DataFrames:
# - usuarios_df: DataFrame con columnas como 'EdadHijos', 'Genero', etc.
# - experiencias_df: DataFrame con columnas como 'EdadHijos_lista', 'Genero', etc.
# - reglas_df: DataFrame con columnas 'atributo' y 'tipo_regla', por ejemplo:
#       | atributo     | tipo_regla |
#       |--------------|------------|
#       | EdadHijos    | rango      |
#       | Genero       | igual      |
#       | TipoMascota  | lista      |

# Puedes generar las condiciones para hacer un join con:
# condiciones = generar_condiciones_join(usuarios_df, experiencias_df, reglas_df)

# Luego, puedes aplicarlas en un filtro o en un join condicional como:
# resultado = usuarios_df.join(experiencias_df, on=condiciones, how="inner")